"""Equivalent retained-model chain kernels with stable target-grouped gathers.

Five contiguous mass/signed-mass channels, contiguous endpoint slices, reusable
message buffers and an optional terminal contraction target inference overhead.
Optional edge aggregation changes summation order, not represented physics.
The selectable legacy scatter kernels retain the earlier implementation;
stable target grouping preserves each target's existing summation order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit, prange, literal_unroll


@njit(cache=True)
def _fht(values):
    half=1;n=len(values)
    while half<n:
        for block in range(0,n,2*half):
            for offset in range(half):
                left,right=block+offset,block+offset+half
                for channel in literal_unroll((0,1,2,3,4)):
                    a,b=values[left,channel],values[right,channel]
                    values[left,channel]=a+b;values[right,channel]=a-b
        half*=2


@njit(cache=True,inline="always")
def _put(values,row,p,b0,b1,b2,b3,scale):
    probability=max(0.,p*scale)
    values[row,0]=probability
    values[row,1]=min(probability,max(-probability,b0*scale))
    values[row,2]=min(probability,max(-probability,b1*scale))
    values[row,3]=min(probability,max(-probability,b2*scale))
    values[row,4]=min(probability,max(-probability,b3*scale))


@njit(cache=True)
def _middle(values,hat,syndrome):
    _fht(values)
    for state in range(len(values)):
        for channel in literal_unroll((0,1,2,3,4)):
            values[state,channel]*=hat[state,channel]
    _fht(values)
    scale=1./len(values)
    if syndrome==0:
        for state in range(len(values)):
            _put(values,state,values[state,0],values[state,1],values[state,2],values[state,3],values[state,4],scale)
    else:
        pivot=1
        while not syndrome&pivot:pivot<<=1
        mask=pivot-1
        for pair in range(len(values)//2):
            state=(pair&mask)|((pair&~mask)<<1);other=state^syndrome
            a0,a1,a2,a3,a4=values[state,0],values[state,1],values[state,2],values[state,3],values[state,4]
            b0,b1,b2,b3,b4=values[other,0],values[other,1],values[other,2],values[other,3],values[other,4]
            _put(values,state,b0,b1,b2,b3,b4,scale)
            _put(values,other,a0,a1,a2,a3,a4,scale)


@njit(cache=True)
def _normalize(values):
    total=0.
    for state in range(len(values)):total+=values[state,0]
    if total>0 and np.isfinite(total):
        for state in range(len(values)):
            for channel in literal_unroll((0,1,2,3,4)):
                values[state,channel]/=total
    return total


@njit(cache=True)
def _first_edge(target,first,prefix,syndrome,left,right,coefficients):
    target[:]=0.
    for path in range(len(left)):
        a=syndrome^left[path];b=right[path]
        for channel in literal_unroll((0,1,2,3,4)):
            target[b,channel]+=coefficients[path,channel]*first[prefix,a,channel]


@njit(cache=True)
def _edge(source,target,left,right,coefficients):
    target[:]=0.
    for path in range(len(left)):
        a,b=left[path],right[path]
        for channel in literal_unroll((0,1,2,3,4)):
            target[b,channel]+=coefficients[path,channel]*source[a,channel]


@njit(cache=True)
def _first_edge_csr(target,first,prefix,syndrome,pointers,left,coefficients):
    """Gather each target in its original relative path summation order."""
    for state in range(len(target)):
        p,b0,b1,b2,b3=0.,0.,0.,0.,0.
        for path in range(pointers[state],pointers[state+1]):
            a=syndrome^left[path]
            p+=coefficients[path,0]*first[prefix,a,0]
            b0+=coefficients[path,1]*first[prefix,a,1]
            b1+=coefficients[path,2]*first[prefix,a,2]
            b2+=coefficients[path,3]*first[prefix,a,3]
            b3+=coefficients[path,4]*first[prefix,a,4]
        target[state,0]=p;target[state,1]=b0;target[state,2]=b1
        target[state,3]=b2;target[state,4]=b3


@njit(cache=True)
def _edge_csr(source,target,pointers,left,coefficients):
    for state in range(len(target)):
        p,b0,b1,b2,b3=0.,0.,0.,0.,0.
        for path in range(pointers[state],pointers[state+1]):
            a=left[path]
            p+=coefficients[path,0]*source[a,0]
            b0+=coefficients[path,1]*source[a,1]
            b1+=coefficients[path,2]*source[a,2]
            b2+=coefficients[path,3]*source[a,3]
            b3+=coefficients[path,4]*source[a,4]
        target[state,0]=p;target[state,1]=b0;target[state,2]=b1
        target[state,3]=b2;target[state,4]=b3


@njit(cache=True)
def _probability_edge_csr(source,target,pointers,left,coefficients):
    for state in range(len(target)):
        probability=0.
        for path in range(pointers[state],pointers[state+1]):
            probability+=coefficients[path,0]*source[left[path]]
        target[state]=probability


@njit(cache=True)
def _tail_projection(source,left):
    projection=np.zeros((left.shape[1],5))
    for state in range(len(source)):
        for rank in range(left.shape[1]):
            for channel in literal_unroll((0,1,2,3,4)):
                projection[rank,channel]+=left[state,rank,channel]*source[state,channel]
    return projection


@njit(cache=True)
def _add_tail_projection(target,right,projection):
    for state in range(len(target)):
        for rank in range(right.shape[1]):
            value=right[state,rank]
            for channel in literal_unroll((0,1,2,3,4)):
                target[state,channel]+=value*projection[rank,channel]


@njit(cache=True)
def _first_tail(target,first,prefix,syndrome,right,left):
    projection=np.zeros((left.shape[1],5))
    for state in range(len(target)):
        for rank in range(left.shape[1]):
            for channel in literal_unroll((0,1,2,3,4)):
                projection[rank,channel]+=left[state,rank,channel]*first[prefix,syndrome^state,channel]
    _add_tail_projection(target,right,projection)


@njit(cache=True)
def _last_dot(current,last,prefix,syndrome):
    result=np.zeros(5)
    for state in range(len(current)):
        index=syndrome^state
        for channel in literal_unroll((0,1,2,3,4)):
            result[channel]+=current[state,channel]*last[prefix,index,channel]
    return result


@njit(cache=True)
def _terminal(current,last,prefix,syndrome,left,right,coefficients):
    result=np.zeros(5)
    for path in range(len(left)):
        a,b=left[path],syndrome^right[path]
        for channel in literal_unroll((0,1,2,3,4)):
            result[channel]+=(coefficients[path,channel]*current[a,channel])*last[prefix,b,channel]
    return result


@njit(cache=True)
def _terminal_tail(current,last,prefix,syndrome,right,left):
    forward=_tail_projection(current,left)
    backward=np.zeros((right.shape[1],5))
    for state in range(len(current)):
        for rank in range(right.shape[1]):
            value=right[state,rank]
            for channel in literal_unroll((0,1,2,3,4)):
                backward[rank,channel]+=value*last[prefix,syndrome^state,channel]
    result=np.zeros(5)
    for rank in range(right.shape[1]):
        for channel in literal_unroll((0,1,2,3,4)):
            result[channel]+=forward[rank,channel]*backward[rank,channel]
    return result


@njit(cache=True)
def _one(source,first,last,middle_hat,left,right,coefficients,tail_right,tail_left,bits,first_prefix,logical_offset,fused_terminal):
    n=1<<bits
    current=np.empty((n,5));other=np.empty((n,5))
    first_low=source[0]&((1<<first_prefix)-1);first_high=source[0]>>first_prefix
    _first_edge(current,first,first_low,first_high,left,right,coefficients)
    if tail_right.shape[1]:_first_tail(current,first,first_low,first_high,tail_right,tail_left)
    valid=_normalize(current)>0
    if valid:
        _middle(current,middle_hat[0],source[1]);valid=_normalize(current)>0
    if valid:
        _edge(current,other,left,right,coefficients)
        if tail_right.shape[1]:_add_tail_projection(other,tail_right,_tail_projection(current,tail_left))
        valid=_normalize(other)>0
    if valid:
        _middle(other,middle_hat[1],source[2]);valid=_normalize(other)>0
    result=np.zeros(5)
    if valid:
        prefix=source[3]>>bits;syndrome=source[3]&((1<<bits)-1)
        if fused_terminal:
            result=_terminal(other,last,prefix,syndrome,left,right,coefficients)
            if tail_right.shape[1]:result+=_terminal_tail(other,last,prefix,syndrome,tail_right,tail_left)
        else:
            _edge(other,current,left,right,coefficients)
            if tail_right.shape[1]:_add_tail_projection(current,tail_right,_tail_projection(other,tail_left))
            valid=_normalize(current)>0
            if valid:result=_last_dot(current,last,prefix,syndrome)
    probability=np.empty(4)
    valid=valid and result[0]>0 and np.isfinite(result[0])
    for bit in range(4):
        if valid:
            correlation=min(1.,max(-1.,result[bit+1]/result[0]))
            if (logical_offset>>bit)&1:correlation=-correlation
            probability[bit]=.5*(1-correlation)
        else:probability[bit]=(logical_offset>>bit)&1
    return probability,valid,result[0]


@njit(cache=True,parallel=True)
def _batch(sources,first,last,middle_hat,left,right,coefficients,tail_right,tail_left,bits,first_prefix,logical_offset,fused_terminal):
    probabilities=np.empty((len(sources),4));valid=np.empty(len(sources),np.bool_);mass=np.empty(len(sources))
    for row in prange(len(sources)):
        probabilities[row],valid[row],mass[row]=_one(sources[row],first,last,middle_hat,left,right,coefficients,tail_right,tail_left,bits,first_prefix,logical_offset,fused_terminal)
    return probabilities,valid,mass


@njit(cache=True)
def _one_csr(source,first,last,middle_hat,left,right,coefficients,tail_right,tail_left,
             pointers,right_left,right_coefficients,bits,first_prefix,logical_offset,fused_terminal):
    n=1<<bits
    current=np.empty((n,5));other=np.empty((n,5))
    first_low=source[0]&((1<<first_prefix)-1);first_high=source[0]>>first_prefix
    _first_edge_csr(current,first,first_low,first_high,pointers,right_left,right_coefficients)
    if tail_right.shape[1]:_first_tail(current,first,first_low,first_high,tail_right,tail_left)
    valid=_normalize(current)>0
    if valid:
        _middle(current,middle_hat[0],source[1]);valid=_normalize(current)>0
    if valid:
        _edge_csr(current,other,pointers,right_left,right_coefficients)
        if tail_right.shape[1]:_add_tail_projection(other,tail_right,_tail_projection(current,tail_left))
        valid=_normalize(other)>0
    if valid:
        _middle(other,middle_hat[1],source[2]);valid=_normalize(other)>0
    result=np.zeros(5)
    if valid:
        prefix=source[3]>>bits;syndrome=source[3]&((1<<bits)-1)
        if fused_terminal:
            # Preserve the original global path order in this reduction.
            result=_terminal(other,last,prefix,syndrome,left,right,coefficients)
            if tail_right.shape[1]:result+=_terminal_tail(other,last,prefix,syndrome,tail_right,tail_left)
        else:
            _edge_csr(other,current,pointers,right_left,right_coefficients)
            if tail_right.shape[1]:_add_tail_projection(current,tail_right,_tail_projection(other,tail_left))
            valid=_normalize(current)>0
            if valid:result=_last_dot(current,last,prefix,syndrome)
    valid=valid and result[0]>0 and np.isfinite(result[0])
    probability=np.empty(4)
    for bit in range(4):
        if valid:
            correlation=min(1.,max(-1.,result[bit+1]/result[0]))
            if (logical_offset>>bit)&1:correlation=-correlation
            probability[bit]=.5*(1-correlation)
        else:probability[bit]=(logical_offset>>bit)&1
    return probability,valid,result[0]


@njit(cache=True,parallel=True)
def _batch_csr(sources,first,last,middle_hat,left,right,coefficients,tail_right,tail_left,
               pointers,right_left,right_coefficients,bits,first_prefix,logical_offset,fused_terminal):
    probability=np.empty((len(sources),4));valid=np.empty(len(sources),np.bool_);mass=np.empty(len(sources))
    for row in prange(len(sources)):
        probability[row],valid[row],mass[row]=_one_csr(sources[row],first,last,middle_hat,left,right,
            coefficients,tail_right,tail_left,pointers,right_left,right_coefficients,bits,first_prefix,
            logical_offset,fused_terminal)
    return probability,valid,mass


def prepare_right_layout(chain):
    """Cache a stable target-grouped view of an immutable retained operator.

    Every target retains its exact original relative path order. No coefficient
    changes; original paths remain available for global terminal reductions.
    """
    identity=(id(chain['left']),id(chain['right']),id(chain['coefficients']))
    if chain.get('_right_layout_identity')==identity:
        return chain['right_ptr'],chain['right_left'],chain['right_coefficients']
    n=1<<chain['bits'];right=chain['right']
    sorting=np.argsort(right,kind='stable')
    counts=np.bincount(np.asarray(right,np.int64),minlength=n)
    chain['right_ptr']=np.ascontiguousarray(np.r_[0,np.cumsum(counts)],dtype=np.int64)
    chain['right_left']=np.ascontiguousarray(chain['left'][sorting])
    chain['right_coefficients']=np.ascontiguousarray(chain['coefficients'][sorting])
    chain['_right_layout_identity']=identity
    return chain['right_ptr'],chain['right_left'],chain['right_coefficients']


def build_fast_chain(chain,*,edge=None,aggregate=False,fused_terminal=True,bits=14,first_prefix=7,residual=None):
    """Accept baseline edge4-tuples or(left,right,coefficients[paths,5])."""
    edge=chain['edge'] if edge is None else edge
    left,right=np.asarray(edge[0],np.uint16),np.asarray(edge[1],np.uint16)
    if len(edge)==4:
        labels,weights=np.asarray(edge[2],np.uint8),np.asarray(edge[3],float)
        coefficients=np.empty((len(left),5));coefficients[:,0]=weights
        for bit in range(4):coefficients[:,bit+1]=weights*np.where((labels>>bit)&1,-1.,1.)
    elif len(edge)==3:coefficients=np.asarray(edge[2],float)
    else:raise ValueError('Edge must have3 or4 arrays')
    if right.shape!=left.shape or coefficients.shape!=(len(left),5) or np.any(left>=(1<<bits)) or np.any(right>=(1<<bits)):
        raise ValueError('Invalid edge dimensions')
    original_paths=len(left)
    if aggregate and len(left):
        key=left.astype(np.uint32)*(1<<bits)+right
        sorting=np.argsort(key,kind='stable');key=key[sorting]
        starts=np.r_[0,np.flatnonzero(key[1:]!=key[:-1])+1]
        coefficients=np.add.reduceat(coefficients[sorting],starts,axis=0)
        left,right=left[sorting[starts]],right[sorting[starts]]
    n=1<<bits;prefixes=1<<first_prefix
    if residual is None:
        tail_right=np.empty((n,0));tail_left=np.empty((n,0,5))
    else:
        tail_right=np.asarray(residual[0],float);left_channels=np.asarray(residual[1],float)
        if tail_right.ndim!=2 or tail_right.shape[0]!=n or left_channels.shape!=(5,tail_right.shape[1],n):
            raise ValueError('Residual must be(right[N,R],left[5,R,N])')
        if np.any(~np.isfinite(tail_right)) or np.any(tail_right<0) or np.any(~np.isfinite(left_channels)):
            raise ValueError('Residual right factor must be nonnegative and all factors finite')
        tail_right=np.ascontiguousarray(tail_right);tail_left=np.ascontiguousarray(left_channels.transpose(2,1,0))
    first=np.empty((prefixes,n,5));last=np.empty((prefixes,n,5))
    first[:,:,0]=chain['first_p'].reshape(n,prefixes).T
    first[:,:,1:]=chain['first_b'].reshape(n,prefixes,4).transpose(1,0,2)
    last[:,:,0]=chain['last_p'].reshape(prefixes,n)
    last[:,:,1:]=chain['last_b'].reshape(prefixes,n,4)
    result={'first':first,'last':last,'middle_hat':np.ascontiguousarray(chain['middle_hat'].transpose(0,2,1)),
            'left':np.ascontiguousarray(left),'right':np.ascontiguousarray(right),'coefficients':np.ascontiguousarray(coefficients),
            'tail_right':tail_right,'tail_left':tail_left,
            'bits':bits,'first_prefix':first_prefix,'fused_terminal':bool(fused_terminal),'regions':chain['regions'],
            'detector_offset':chain['detector_offset'],'logical_offset':chain['logical_offset'],
            'metadata':{'original_paths':original_paths,'active_paths':len(left),'aggregated':bool(aggregate),
                        'fused_terminal':bool(fused_terminal),'residual_rank':tail_right.shape[1],
                        'model_support_unchanged':edge is chain['edge'] and tail_right.shape[1]==0}}
    prepare_right_layout(result)
    return result


def decode_fast_chain(chain,events,*,deduplicate=True,kernel='csr'):
    if kernel not in ('csr','legacy'):raise ValueError("kernel must be 'csr' or 'legacy'")
    events=np.asarray(events)
    detectors=len(chain['detector_offset'])
    if events.ndim!=2 or events.shape[1]!=detectors or not np.all((events==0)|(events==1)):
        raise ValueError('Expected binary detector-event matrix')
    events=np.asarray(events,np.uint8)
    inverse=None
    if deduplicate and len(events):
        packed=np.ascontiguousarray(np.packbits(events,axis=1))
        keys=packed.view(np.dtype((np.void,packed.shape[1]))).ravel()
        _,first,inverse=np.unique(keys,return_index=True,return_inverse=True)
        events=np.ascontiguousarray(events[first])
    normalized=events^chain['detector_offset']
    sources=np.column_stack([normalized[:,r]@(1<<np.arange(len(r))) for r in chain['regions']]).astype(np.int64)
    partial_info={}
    if kernel=='csr':
        if chain.get('partial_cache') is not None:
            from diagnostics.tensor_partial_cache import decode_partial_sources
            probability,valid,mass,partial_info=decode_partial_sources(chain,sources)
        else:
            pointers,right_left,right_coefficients=prepare_right_layout(chain)
            probability,valid,mass=_batch_csr(sources,chain['first'],chain['last'],chain['middle_hat'],chain['left'],chain['right'],
                chain['coefficients'],chain['tail_right'],chain['tail_left'],pointers,right_left,right_coefficients,
                chain['bits'],chain['first_prefix'],chain['logical_offset'],chain['fused_terminal'])
    else:
        probability,valid,mass=_batch(sources,chain['first'],chain['last'],chain['middle_hat'],chain['left'],chain['right'],
            chain['coefficients'],chain['tail_right'],chain['tail_left'],chain['bits'],chain['first_prefix'],chain['logical_offset'],chain['fused_terminal'])
    unique=len(events)
    if inverse is not None:probability,valid,mass=probability[inverse],valid[inverse],mass[inverse]
    return probability,{'valid':valid,'terminal_contraction_mass':mass,'unique_syndromes':unique,'edge_kernel':kernel,
        **chain['metadata'],**partial_info}


def main(argv=None):
    import numba,stim
    import probability_decoder
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots-per-basis',type=int,default=10000)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--report',type=Path,default=root/'reports/target03/tensor_fast_chain.json')
    args=parser.parse_args(argv);numba.set_num_threads(args.threads)
    if args.report.exists():parser.error('Report exists; select a new path')
    frozen=['probability_decoder.py','diagnostics/temporal_chain.py','data/target03/boundary21_X.npz','data/target03/boundary21_Z.npz']
    before={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in frozen}
    report={'purpose':'Equivalent retained-model speed prototypes; no frozen-source changes','threads':args.threads,'per_basis':{},'complete':False,'frozen_sha256_before':before}
    with np.load(root/'data/target03/probability_decoder_fresh100k.npz',allow_pickle=False) as data:
        for basis in 'XZ':
            core=probability_decoder.get_decoder(stim.Circuit.from_file(str(root/f'data/competition/{basis}_circuit.stim')))
            events=data[basis+'_events'][:args.shots_per_basis];reference=data[basis+'_probabilities'][:args.shots_per_basis]
            report['per_basis'][basis]={}
            for name,aggregate,fused in (('aos',False,False),('aos_terminal',False,True),('aos_aggregated_terminal',True,True)):
                fast=build_fast_chain(core.chain,aggregate=aggregate,fused_terminal=fused)
                core.predict_proba(events[:2]);decode_fast_chain(fast,events[:2])
                timings={'baseline':[],'candidate':[]};candidate=None
                for pair in range(2):
                    order=('baseline','candidate') if pair==0 else ('candidate','baseline')
                    for arm in order:
                        started=time.perf_counter()
                        output=core.predict_proba(events) if arm=='baseline' else decode_fast_chain(fast,events)[0]
                        timings[arm].append(time.perf_counter()-started)
                        if arm=='candidate':candidate=output
                difference=np.abs(candidate-reference);changed=np.count_nonzero((candidate>.5)!=(reference>.5))
                row={'max_probability_delta':float(difference.max(initial=0)),'changed_prediction_bits':int(changed),
                     'shots':len(events),'timings':timings,'candidate_over_baseline':float(np.mean(timings['candidate'])/np.mean(timings['baseline'])),
                     'metadata':fast['metadata']}
                report['per_basis'][basis][name]=row
                print(basis,name,json.dumps(row),flush=True)
                del fast
    report['frozen_sha256_after']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in frozen}
    report['inputs_unchanged']=before==report['frozen_sha256_after'];report['complete']=True
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':main()
