"""Exact selected-left-row edge additions to the retained temporal chain.

The existing sparse operator is always evaluated in full. ``delta[k,b,c]``
adds omitted probability/signed-logical mass from left state ``selected[k]``
to right state ``b``. A fixed set of left rows therefore defines one positive
edge model; these same rows are used at all three edge occurrences.

This is an experimental runtime. It does not alter the frozen decoder.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange, literal_unroll

from diagnostics.tensor_fast_chain import (
    build_fast_chain, decode_fast_chain, _first_edge, _edge, _middle,
    _normalize, _terminal, _last_dot,
)


@njit(cache=True)
def _gather_selected(source, selected, count):
    values=np.empty((count,5))
    for row in range(count):
        state=selected[row]
        for channel in literal_unroll((0,1,2,3,4)):
            values[row,channel]=source[state,channel]
    return values


@njit(cache=True)
def _gather_first(first,prefix,syndrome,selected,count):
    values=np.empty((count,5))
    for row in range(count):
        state=syndrome^selected[row]
        for channel in literal_unroll((0,1,2,3,4)):
            values[row,channel]=first[prefix,state,channel]
    return values


@njit(cache=True)
def _add_rows(target,selected_values,delta,count):
    """Five independent dense matvecs, contiguous within each (b,k)."""
    for state in range(len(target)):
        p,b0,b1,b2,b3=0.,0.,0.,0.,0.
        for row in range(count):
            p+=delta[state,row,0]*selected_values[row,0]
            b0+=delta[state,row,1]*selected_values[row,1]
            b1+=delta[state,row,2]*selected_values[row,2]
            b2+=delta[state,row,3]*selected_values[row,3]
            b3+=delta[state,row,4]*selected_values[row,4]
        target[state,0]+=p
        target[state,1]+=b0
        target[state,2]+=b1
        target[state,3]+=b2
        target[state,4]+=b3


@njit(cache=True)
def _terminal_rows(selected_values,delta,count,last,prefix,syndrome):
    result=np.zeros(5)
    for state in range(len(delta)):
        p,b0,b1,b2,b3=0.,0.,0.,0.,0.
        for row in range(count):
            p+=delta[state,row,0]*selected_values[row,0]
            b0+=delta[state,row,1]*selected_values[row,1]
            b1+=delta[state,row,2]*selected_values[row,2]
            b2+=delta[state,row,3]*selected_values[row,3]
            b3+=delta[state,row,4]*selected_values[row,4]
        index=syndrome^state
        result[0]+=p*last[prefix,index,0]
        result[1]+=b0*last[prefix,index,1]
        result[2]+=b1*last[prefix,index,2]
        result[3]+=b2*last[prefix,index,3]
        result[4]+=b3*last[prefix,index,4]
    return result


@njit(cache=True)
def _one(source,first,last,middle_hat,left,right,coefficients,selected,delta,
         count,bits,first_prefix,logical_offset,fused_terminal):
    n=1<<bits
    current=np.empty((n,5));other=np.empty((n,5))
    first_low=source[0]&((1<<first_prefix)-1)
    first_high=source[0]>>first_prefix
    _first_edge(current,first,first_low,first_high,left,right,coefficients)
    _add_rows(current,_gather_first(first,first_low,first_high,selected,count),delta,count)
    valid=_normalize(current)>0
    if valid:
        _middle(current,middle_hat[0],source[1]);valid=_normalize(current)>0
    if valid:
        _edge(current,other,left,right,coefficients)
        _add_rows(other,_gather_selected(current,selected,count),delta,count)
        valid=_normalize(other)>0
    if valid:
        _middle(other,middle_hat[1],source[2]);valid=_normalize(other)>0
    result=np.zeros(5)
    if valid:
        prefix=source[3]>>bits;syndrome=source[3]&((1<<bits)-1)
        selected_values=_gather_selected(other,selected,count)
        if fused_terminal:
            result=_terminal(other,last,prefix,syndrome,left,right,coefficients)
            result+=_terminal_rows(selected_values,delta,count,last,prefix,syndrome)
        else:
            _edge(other,current,left,right,coefficients)
            _add_rows(current,selected_values,delta,count)
            valid=_normalize(current)>0
            if valid:result=_last_dot(current,last,prefix,syndrome)
    probabilities=np.empty(4)
    valid=valid and result[0]>0 and np.isfinite(result[0])
    for bit in range(4):
        if valid:
            correlation=min(1.,max(-1.,result[bit+1]/result[0]))
            if (logical_offset>>bit)&1:correlation=-correlation
            probabilities[bit]=.5*(1-correlation)
        else:probabilities[bit]=(logical_offset>>bit)&1
    return probabilities,valid,result[0]


@njit(cache=True,parallel=True)
def _batch(sources,first,last,middle_hat,left,right,coefficients,selected,delta,
           count,bits,first_prefix,logical_offset,fused_terminal):
    probabilities=np.empty((len(sources),4))
    valid=np.empty(len(sources),np.bool_);mass=np.empty(len(sources))
    for row in prange(len(sources)):
        probabilities[row],valid[row],mass[row]=_one(sources[row],first,last,
            middle_hat,left,right,coefficients,selected,delta,count,bits,
            first_prefix,logical_offset,fused_terminal)
    return probabilities,valid,mass


def build_sliced_chain(chain,selected_left=None,delta=None,*,tail_layout='KN5',
                       maximum_rows=None,aggregate=True,fused_terminal=True,
                       bits=14,first_prefix=7):
    """Build a chain with additional exact left rows, preserving all old paths.

    ``chain`` may be a frozen-format chain or an already built fast chain.
    Input delta is [K,N,5] by default, or [N,K,5] with tail_layout='NK5'.
    Channels are probability followed by four signed logical-bit masses.
    The row order is retained, so ``maximum_rows`` chooses an exact prefix.
    Existing low-rank residuals are rejected rather than silently dropped.
    """
    fast=chain if 'coefficients' in chain and 'first' in chain else build_fast_chain(
        chain,aggregate=aggregate,fused_terminal=fused_terminal,bits=bits,
        first_prefix=first_prefix)
    if fast.get('tail_right',np.empty((0,0))).shape[1]:
        raise ValueError('Combine low-rank and selected-row tails explicitly before building')
    if (selected_left is None)!=(delta is None):
        raise ValueError('Supply both selected_left and delta, or neither')
    n=1<<fast['bits']
    if selected_left is None:
        selected=np.empty(0,np.int64);packed=np.empty((n,0,5))
    else:
        raw=np.asarray(selected_left)
        if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.integer):
            raise ValueError('Selected left states must be a one-dimensional integer array')
        selected=np.asarray(raw,np.int64)
        if np.any(selected<0) or np.any(selected>=n) or len(np.unique(selected))!=len(selected):
            raise ValueError('Selected left states must be unique and in range')
        values=np.asarray(delta,dtype=float)
        if tail_layout=='KN5':
            if values.shape!=(len(selected),n,5):raise ValueError('Expected delta[K,N,5]')
            values=values.transpose(1,0,2)
        elif tail_layout=='NK5':
            if values.shape!=(n,len(selected),5):raise ValueError('Expected delta[N,K,5]')
        else:raise ValueError("tail_layout must be 'KN5' or 'NK5'")
        # Permit small Fourier/subtraction roundoff without changing coefficients.
        # Validate bounded blocks: the real mmap is640MiB and full-array abs()
        # temporaries would otherwise create another half-GiB memory peak.
        tolerance=1e-12
        for start in range(0,n,64):
            block=values[start:start+64]
            if np.any(~np.isfinite(block)):
                raise ValueError('Tail coefficients must be finite')
            if np.any(block[:,:,0]<-tolerance) or np.any(np.abs(block[:,:,1:])>block[:,:,0,None]+tolerance):
                raise ValueError('Tail signed masses exceed nonnegative probability mass')
        packed=np.ascontiguousarray(values)
    if maximum_rows is None:maximum_rows=len(selected)
    if not isinstance(maximum_rows,(int,np.integer)) or not 0<=maximum_rows<=len(selected):
        raise ValueError('maximum_rows must select a prefix of available rows')
    result=dict(fast)
    result['selected_left']=np.ascontiguousarray(selected)
    result['row_delta']=packed
    result['maximum_rows']=int(maximum_rows)
    result['metadata']={**fast['metadata'],'selected_rows_available':len(selected),
        'selected_rows':int(maximum_rows),'tail_layout':'NK5',
        'tail_bytes':int(packed.nbytes+selected.nbytes),
        'model_support_unchanged':maximum_rows==0 and fast['metadata'].get('model_support_unchanged',False)}
    return result


def _events(chain,events):
    values=np.asarray(events)
    if values.ndim!=2 or values.shape[1]!=len(chain['detector_offset']) or not np.all((values==0)|(values==1)):
        raise ValueError('Expected binary detector-event matrix')
    return np.asarray(values,np.uint8)


def _decode_selected(chain,events,count,deduplicate):
    inverse=None
    if deduplicate and len(events):
        packed=np.ascontiguousarray(np.packbits(events,axis=1))
        keys=packed.view(np.dtype((np.void,packed.shape[1]))).ravel()
        _,first,inverse=np.unique(keys,return_index=True,return_inverse=True)
        events=np.ascontiguousarray(events[first])
    normalized=events^chain['detector_offset']
    sources=np.column_stack([normalized[:,r]@(1<<np.arange(len(r))) for r in chain['regions']]).astype(np.int64)
    probabilities,valid,mass=_batch(sources,chain['first'],chain['last'],
        chain['middle_hat'],chain['left'],chain['right'],chain['coefficients'],
        chain['selected_left'],chain['row_delta'],count,chain['bits'],
        chain['first_prefix'],chain['logical_offset'],chain['fused_terminal'])
    unique=len(events)
    if inverse is not None:probabilities,valid,mass=probabilities[inverse],valid[inverse],mass[inverse]
    return probabilities,valid,mass,unique


def decode_sliced_chain(chain,events,*,maximum_rows=None,gate_mask=None,
                        minimum_uncertainty=None,reference_probabilities=None,
                        deduplicate=True):
    """Contract all selected rows, optionally only for a fixed baseline gate.

    A probability gate uses max_bit min(q,1-q)>=minimum_uncertainty. If no
    reference is supplied, the fast retained-model baseline is computed once.
    Explicit gate_mask and minimum_uncertainty are mutually exclusive. Rejected
    shots retain their reference probabilities exactly. With no gate all shots
    use the same extended edge model; labels never participate in inference.
    """
    events=_events(chain,events)
    count=chain['maximum_rows'] if maximum_rows is None else maximum_rows
    if not isinstance(count,(int,np.integer)) or not 0<=count<=len(chain['selected_left']):
        raise ValueError('maximum_rows must select a prefix of available rows')
    if gate_mask is not None and minimum_uncertainty is not None:
        raise ValueError('Choose one gate definition')
    reference=None
    if reference_probabilities is not None:
        reference=np.asarray(reference_probabilities,float)
        if reference.shape!=(len(events),4) or np.any(~np.isfinite(reference)) or np.any((reference<0)|(reference>1)):
            raise ValueError('Expected finite reference probabilities[n,4] in[0,1]')
    gated=gate_mask is not None or minimum_uncertainty is not None
    baseline_details=None
    if gated or count==0:
        if reference is None:reference,baseline_details=decode_fast_chain(chain,events,deduplicate=deduplicate)
    if gate_mask is not None:
        mask=np.asarray(gate_mask)
        if mask.shape!=(len(events),) or mask.dtype!=np.bool_:
            raise ValueError('gate_mask must be a boolean vector matching events')
    elif minimum_uncertainty is not None:
        if not np.isfinite(minimum_uncertainty) or not 0<=minimum_uncertainty<=.5:
            raise ValueError('minimum_uncertainty must be in[0,.5]')
        mask=np.max(np.minimum(reference,1-reference),axis=1)>=minimum_uncertainty
    else:mask=np.ones(len(events),np.bool_)
    if count==0:
        mask=np.zeros(len(events),np.bool_)
    probabilities=reference.copy() if reference is not None else np.empty((len(events),4))
    valid=np.ones(len(events),np.bool_)
    mass=np.full(len(events),np.nan)
    if baseline_details is not None:
        valid[:]=baseline_details['valid'];mass[:]=baseline_details['terminal_contraction_mass']
    unique=0
    if np.any(mask):
        probability,tail_valid,tail_mass,unique=_decode_selected(chain,events[mask],int(count),deduplicate)
        probabilities[mask]=probability;valid[mask]=tail_valid;mass[mask]=tail_mass
    return probabilities,{'valid':valid,'terminal_contraction_mass':mass,
        'tail_applied':mask,'tail_shots':int(mask.sum()),'tail_unique_syndromes':unique,
        **chain['metadata'],'selected_rows':int(count)}
