"""One-omitted-edge tensor contraction using fixed five-channel environments.

Z = Z_reference + sum_edge F_edge Delta_edge B_edge. Terms with two or
three omitted edges are excluded. Selected row corrections preserve their
signed logical masses. A remaining-row bound is conditional on this one-tail
model; the unconditional multiple-tail event probability is a separate fact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit, prange

from diagnostics.tensor_fast_chain import _edge, _middle, _normalize


def transpose_tail_file(archive_path, *, block_rows=64):
    """Create/reuse a disk-backed K-major copy without a full in-RAM transpose."""
    archive_path=Path(archive_path)
    with np.load(archive_path,allow_pickle=False) as archive:
        selected=archive['selected_left'].copy();metadata=json.loads(str(archive['metadata']))
    source_path=archive_path.parent/metadata['coefficient_file']
    output_path=source_path.with_name(source_path.stem+'_KN5.npy')
    companion=output_path.with_suffix('.npz')
    if output_path.exists() or companion.exists():
        if not(output_path.exists() and companion.exists()):
            raise ValueError('Incomplete K-major companion; select a fresh output name')
        with np.load(companion,allow_pickle=False) as cached:
            info=json.loads(str(cached['metadata']))
            if info['source_metadata']!=metadata or not np.array_equal(cached['selected_left'],selected):
                raise ValueError('K-major companion identity mismatch')
        matrix=np.load(output_path,mmap_mode='r')
        if matrix.shape!=(len(selected),1<<metadata['bits_per_side'],5) or matrix.dtype!=np.float64:
            raise ValueError('Malformed K-major companion')
        return selected,matrix,info
    source=np.load(source_path,mmap_mode='r')
    n,k,c=source.shape
    if k!=len(selected) or c!=5 or source.dtype!=np.float64 or block_rows<1:
        raise ValueError('Unexpected selected-edge archive layout')
    output=np.lib.format.open_memmap(output_path,mode='w+',dtype=np.float64,shape=(k,n,5))
    payload=hashlib.sha256();row_mass=np.empty(k)
    for start in range(0,k,block_rows):
        stop=min(k,start+block_rows)
        block=np.ascontiguousarray(source[:,start:stop,:].transpose(1,0,2))
        if not np.isfinite(block).all():raise ValueError('Nonfinite selected-edge coefficient')
        output[start:stop]=block;row_mass[start:stop]=block[:,:,0].sum(axis=1)
        payload.update(memoryview(block).cast('B'));output.flush()
        del block
    output.flush();del output,source
    info={'source_metadata':metadata,'source_archive':str(archive_path),
        'layout':'K,N,5','coefficient_file':output_path.name,'payload_sha256':payload.hexdigest(),
        'transpose_block_rows':block_rows,'copy_only_no_inference_change':True}
    np.savez_compressed(companion,selected_left=selected,row_mass=row_mass,metadata=np.asarray(json.dumps(info)))
    return selected,np.load(output_path,mmap_mode='r'),info


def omitted_row_masses(left,probabilities,reference_left,reference_weights,bits=14):
    """All-order left marginal minus retained paths, with raw roundoff ledger."""
    from diagnostics.tensor_full_edge import left_marginal
    all_mass=left_marginal(np.asarray(left,np.uint16),np.asarray(probabilities,float),bits)
    retained=np.bincount(reference_left,weights=reference_weights,minlength=1<<bits)
    raw=all_mass-retained
    return np.maximum(raw,0.),{'negative_roundoff_sum':float(-raw[raw<0].sum()),
        'negative_roundoff_max':float(max(0.,-raw.min(initial=0))),
        'full_edge_mass':float(all_mass.sum()),'retained_edge_mass':float(retained.sum()),
        'bound_note':'Mathematical bound evaluated in float64; not an interval-arithmetic certificate.'}


@njit(cache=True)
def edge_adjoint(source,left,right,coefficients):
    output=np.zeros_like(source)
    for path in range(len(left)):
        a,b=left[path],right[path]
        for channel in range(5):output[a,channel]+=coefficients[path,channel]*source[b,channel]
    return output


@njit(cache=True)
def five_environments(sources,first,last,middle_hat,left,right,coefficients,bits=14,first_prefix=7):
    n=1<<bits;forward=np.empty((3,n,5));backward=np.empty((3,n,5))
    prefix=sources[0]&((1<<first_prefix)-1);syndrome=sources[0]>>first_prefix
    for state in range(n):
        for channel in range(5):forward[0,state,channel]=first[prefix,syndrome^state,channel]
    valid=_normalize(forward[0])>0
    for edge in range(2):
        _edge(forward[edge],forward[edge+1],left,right,coefficients)
        valid=valid and _normalize(forward[edge+1])>0
        _middle(forward[edge+1],middle_hat[edge],sources[edge+1])
        valid=valid and _normalize(forward[edge+1])>0
    prefix=sources[3]>>bits;syndrome=sources[3]&((1<<bits)-1)
    for state in range(n):
        for channel in range(5):backward[2,state,channel]=last[prefix,syndrome^state,channel]
    valid=valid and _normalize(backward[2])>0
    for edge in range(1,-1,-1):
        current=edge_adjoint(backward[edge+1],left,right,coefficients)
        valid=valid and _normalize(current)>0
        _middle(current,middle_hat[edge],sources[edge+1])
        valid=valid and _normalize(current)>0
        backward[edge]=current
    denominators=np.zeros(3)
    for edge in range(3):
        for path in range(len(left)):
            denominators[edge]+=forward[edge,left[path],0]*coefficients[path,0]*backward[edge,right[path],0]
        valid=valid and denominators[edge]>0 and np.isfinite(denominators[edge])
    return forward,backward,denominators,valid


@njit(cache=True)
def one_tail_one(sources,reference,first,last,middle_hat,left,right,coefficients,selected,
                 tail_kn5,row_mass,maximum_rows,remaining_tolerance,logical_offset,
                 bits=14,first_prefix=7):
    forward,backward,denominators,valid=five_environments(sources,first,last,middle_hat,
        left,right,coefficients,bits,first_prefix)
    correction=np.zeros(5);counts=np.zeros(3,np.int32);remaining=np.zeros(3)
    evaluated=np.zeros((3,len(selected)),np.bool_)
    bounds=np.zeros((3,len(selected)))
    if not valid:return reference.copy(),counts,remaining,correction,False
    for edge in range(3):
        maximum=backward[edge,:,0].max()
        for a in range(len(row_mass)):
            remaining[edge]+=forward[edge,a,0]*row_mass[a]*maximum/denominators[edge]
        for index in range(len(selected)):
            a=selected[index]
            bounds[edge,index]=forward[edge,a,0]*row_mass[a]*maximum/denominators[edge]
    # Greedy choice across all three occurrences, each with its own row cap.
    for _ in range(3*maximum_rows):
        if remaining.sum()<=remaining_tolerance:break
        best_edge,best_index,best_score=-1,-1,0.
        for edge in range(3):
            if counts[edge]>=maximum_rows:continue
            for index in range(len(selected)):
                if not evaluated[edge,index] and bounds[edge,index]>best_score:
                    best_edge,best_index,best_score=edge,index,bounds[edge,index]
        if best_edge<0:break
        edge,index=best_edge,best_index;a=selected[index]
        value=np.zeros(5)
        for b in range(tail_kn5.shape[1]):
            for channel in range(5):
                value[channel]+=tail_kn5[index,b,channel]*backward[edge,b,channel]
        for channel in range(5):
            correction[channel]+=forward[edge,a,channel]*value[channel]/denominators[edge]
        evaluated[edge,index]=True;counts[edge]+=1
        remaining[edge]=max(0.,remaining[edge]-best_score)
    q=np.empty(4);normalizer=1+correction[0]
    valid=valid and normalizer>0 and np.isfinite(normalizer)
    for bit in range(4):
        correlation=1-2*reference[bit]
        # Environment characters are in normalized-model coordinates.
        if(logical_offset>>bit)&1:correlation=-correlation
        correlation=(correlation+correction[bit+1])/normalizer
        if(logical_offset>>bit)&1:correlation=-correlation
        q[bit]=.5*(1-min(1.,max(-1.,correlation))) if valid else reference[bit]
    return q,counts,remaining,correction,valid


@njit(cache=True,parallel=True)
def _batch(sources,reference,gate,first,last,middle_hat,left,right,coefficients,selected,
           tail_kn5,row_mass,maximum_rows,tolerance,fallback_unbounded,logical_offset,bits,first_prefix):
    q=reference.copy();counts=np.zeros((len(q),3),np.int32);remaining=np.zeros((len(q),3))
    correction=np.zeros((len(q),5));valid=np.ones(len(q),np.bool_);fallback=np.zeros(len(q),np.bool_)
    for row in prange(len(q)):
        if not gate[row]:continue
        q[row],counts[row],remaining[row],correction[row],valid[row]=one_tail_one(sources[row],
            reference[row],first,last,middle_hat,left,right,coefficients,selected,tail_kn5,row_mass,
            maximum_rows,tolerance,logical_offset,bits,first_prefix)
        if not valid[row] or(fallback_unbounded and remaining[row].sum()>tolerance):
            q[row]=reference[row];fallback[row]=True
    return q,counts,remaining,correction,valid,fallback


def decode_environment_tail(fast,events,reference_probabilities,selected_left,tail_kn5,row_mass,
                            *,maximum_rows=64,remaining_tolerance=1e-4,minimum_uncertainty=.4,
                            fallback_unbounded=True):
    events=np.asarray(events);reference=np.asarray(reference_probabilities,float)
    selected=np.asarray(selected_left,np.int64);mass=np.asarray(row_mass,float)
    n=1<<fast['bits'];tail=np.asarray(tail_kn5)
    if events.ndim!=2 or events.shape[1]!=len(fast['detector_offset']) or not np.all((events==0)|(events==1)):
        raise ValueError('Expected binary detector events')
    if reference.shape!=(len(events),4) or not np.isfinite(reference).all() or np.any((reference<0)|(reference>1)):
        raise ValueError('Expected finite reference probabilities[n,4]')
    if tail.shape!=(len(selected),n,5) or tail.dtype!=np.float64 or not tail.flags.c_contiguous:
        raise ValueError('Expected contiguous float64 K,N,5 coefficients, preferably a mmap')
    if mass.shape!=(n,) or np.any(mass<0) or not np.isfinite(mass).all():raise ValueError('Need nonnegative all-row tail masses')
    if selected.ndim!=1 or len(np.unique(selected))!=len(selected) or np.any((selected<0)|(selected>=n)):
        raise ValueError('Selected left states must be unique and valid')
    if not isinstance(maximum_rows,(int,np.integer)) or maximum_rows<0 or remaining_tolerance<0 or not 0<=minimum_uncertainty<=.5:
        raise ValueError('Invalid truncation controls')
    normalized=events.astype(np.uint8)^fast['detector_offset']
    sources=np.column_stack([normalized[:,r]@(1<<np.arange(len(r))) for r in fast['regions']]).astype(np.int64)
    gate=np.max(np.minimum(reference,1-reference),axis=1)>=minimum_uncertainty
    result=_batch(sources,reference,gate,fast['first'],fast['last'],fast['middle_hat'],fast['left'],
        fast['right'],fast['coefficients'],selected,tail,mass,maximum_rows,remaining_tolerance,
        fallback_unbounded,fast['logical_offset'],fast['bits'],fast['first_prefix'])
    q,counts,remaining,correction,valid,fallback=result
    conditional_bound=remaining.sum(axis=1)/(1+correction[:,0])
    return q,{'gate':gate,'selected_rows':counts,'remaining_relative_mass_bound':remaining,
        'conditional_one_tail_probability_error_bound':conditional_bound,'relative_mass_correction':correction,
        'valid':valid,'fallback':fallback,'bound_note':'Float64 mathematical bound for unevaluated single-tail rows only. Multiple-tail events and Fourier roundoff are separate omissions.'}
