"""Exact two-component fixed-point QUBO minimization.

The reduced-grid CPU solver finds the global finite-grid optimum by
conditioning on one scalar. Full binary enumeration is optional on CUDA.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ExactResult:
    bitstring: str
    decoded: tuple[float, float]
    residual_squared: float
    second_bitstring: str
    second_residual_squared: float
    distinct_minimizer_count: int
    examined: int
    method: str


def codes_to_bits(code: int, M: int, K: int) -> np.ndarray:
    offset = (2**M) * (2**K)
    if not -offset <= code < offset:
        raise ValueError('code outside signed fixed-point range')
    remainder = code + offset if code < 0 else code
    bits = [int(code < 0)]
    bits.extend((remainder >> (K+i)) & 1 for i in range(M))
    bits.extend((remainder >> (K-i)) & 1 for i in range(1, K+1))
    return np.array(bits, dtype=np.uint8)


def decode_bits(bits: np.ndarray | str, gamma: float, M: int=1, K: int=10) -> np.ndarray:
    a=np.array(list(map(int,bits)), dtype=float)
    q=1+M+K
    if a.size != 2*q or not np.all((a==0)|(a==1)):
        raise ValueError('expected two fixed-point scalars')
    weights=np.array([-(2.**M)]+[2.**i for i in range(M)]+[2.**-i for i in range(1,K+1)])*gamma
    return np.array([a[:q]@weights,a[q:]@weights])


def canonical_bits(code0: int, code1: int, M: int, K: int) -> str:
    return ''.join(map(str,np.concatenate((codes_to_bits(code0,M,K),codes_to_bits(code1,M,K)))))


def qubo_upper(S: np.ndarray, column: int, gamma: float, M: int=1,K: int=10) -> np.ndarray:
    S=np.asarray(S,float)
    if S.shape != (2,2) or not 0 <= column <2: raise ValueError('B=2, column 0 or 1')
    q=1+M+K; weights=gamma*np.array([-(2.**M)]+[2.**i for i in range(M)]+[2.**-i for i in range(1,K+1)])
    P=np.zeros((2,2*q)); P[0,:q]=weights; P[1,q:]=weights
    AP=S@P; H=AP.T@AP; lin=-2*AP.T@np.eye(2)[:,column]
    Q=np.triu(2*H,1)+np.diag(np.diag(H)+lin)
    return Q


def reduced_exact(S: np.ndarray, target: np.ndarray, gamma:float, M:int=1,K:int=10) -> ExactResult:
    """Exact global two-scalar minimization, using exact nearest-grid reduction.

    For each first code, second-code objective is a positive quadratic; its
    grid optimum is one of floor/ceil of its continuous optimum. For the
    second distinct best state, also include one neighbor on either side.
    This is a proof of optimality over every representable pair, not a
    heuristic. Numbers are evaluated in float64; very close ties must be
    checked using additional precision if a mathematical certificate is
    required beyond floating-point accuracy.
    """
    S=np.asarray(S,dtype=float); target=np.asarray(target,dtype=float)
    if S.shape!=(2,2) or target.shape!=(2,) or gamma<=0 or not np.all(np.isfinite(S)):
        raise ValueError('invalid Schur inputs')
    if np.linalg.matrix_rank(S)<2: raise ValueError('Schur block must be nonsingular')
    scale=(2**K)/gamma; bound=(2**M)*(2**K)
    xcode=np.arange(-bound,bound,dtype=np.int64)
    xval=xcode/scale
    v=S[:,0,None]*xval[None,:]-target[:,None]
    sy=S[:,1]; denom=float(sy@sy)
    continuous=-(sy@v)/denom*scale
    flo=np.floor(continuous).astype(np.int64)
    energies=[]; ranks=[]; xc=[]; zz=[]
    q=1+M+K
    def as_bit_integer(codes):
        rem=np.where(codes<0,codes+bound,codes)
        value=(codes<0).astype(np.int64)<<(q-1)
        for j in range(M):
            value+=((rem>>(K+j))&1)<<(q-2-j)
        return value+(rem&((1<<K)-1))
    xrank=as_bit_integer(xcode)
    for shift in (-1,0,1,2):
        yc=np.clip(flo+shift,-bound,bound-1)
        rr=v+sy[:,None]*(yc/scale)[None,:]
        energies.append(np.sum(rr*rr,axis=0))
        ranks.append((xrank<<q)+as_bit_integer(yc))
        xc.append(xcode);zz.append(yc)
    energy=np.concatenate(energies);ranks=np.concatenate(ranks)
    xcs=np.concatenate(xc);ycs=np.concatenate(zz)
    order=np.lexsort((ranks,energy))
    first_idx=int(order[0]);second_idx=next(int(i) for i in order[1:] if ranks[i]!=ranks[first_idx])
    i,j=first_idx,second_idx
    fb=canonical_bits(int(xcs[i]),int(ycs[i]),M,K)
    sb=canonical_bits(int(xcs[j]),int(ycs[j]),M,K)
    minimum=float(energy[i])
    count=int(np.unique(ranks[abs(energy-minimum)<=(1e-13*max(1.,minimum))]).size)
    return ExactResult(fb,tuple(float(x) for x in decode_bits(fb,gamma,M,K)),minimum,
                       sb,float(energy[j]),count,int((2*bound)**2),'cpu_reduced_grid')

def full_binary_exact(S:np.ndarray,target:np.ndarray,gamma:float,M:int=1,K:int=10,chunk_power:int=16,device:str='cuda') -> ExactResult:
    """Independent full-bitspace torch enumeration (CUDA or CPU smoke test)."""
    try: import torch
    except ImportError as exc: raise RuntimeError('install torch with CUDA') from exc
    if device not in ('cuda','cpu'): raise ValueError('device must be cuda or cpu')
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('a CUDA GPU is required for full GPU enumeration')
    n=2*(1+M+K)
    if n>30: raise ValueError('exhaustive GPU oracle restricted to <=30 bits')
    if chunk_power<1 or chunk_power>20:raise ValueError('invalid chunk')
    s=torch.tensor(np.asarray(S,float),device=device,dtype=torch.float64)
    target=torch.tensor(np.asarray(target,float),device=device,dtype=torch.float64)
    w=torch.tensor(gamma*np.array([-(2.**M)]+[2.**i for i in range(M)]+[2.**-i for i in range(1,K+1)]),device=device,dtype=torch.float64)
    total=1<<n; shifts=torch.arange(n-1,-1,-1,device=device,dtype=torch.int64)
    best=[]
    for begin in range(0,total,1<<chunk_power):
        codes=torch.arange(begin,min(total,begin+(1<<chunk_power)),device=device,dtype=torch.int64)
        bits=((codes[:,None]>>shifts[None,:])&1).to(torch.float64)
        q=1+M+K;y=torch.stack((bits[:,:q]@w,bits[:,q:]@w),dim=1)
        res=y@s.T-target[None,:]
        obj=(res*res).sum(1)
        vals,ix=torch.topk(obj,k=2,largest=False,sorted=True)
        for val,index in zip(vals.cpu().tolist(),ix.cpu().tolist()):
            best.append((val,int(codes[index].item())))
        # Keep best two candidates globally to avoid memory growth
        best=sorted(best,key=lambda z:(z[0],z[1]))[:2]
    first,second=best
    fb=format(first[1],f'0{n}b');sb=format(second[1],f'0{n}b')
    return ExactResult(fb,tuple(map(float,decode_bits(fb,gamma,M,K))),float(first[0]),sb,float(second[0]),0,total,f'{device}_full_binary_enumeration')


def cuda_full_exact(S,target,gamma,M=1,K=10,chunk_power=16):
    return full_binary_exact(S,target,gamma,M,K,chunk_power,device='cuda')
