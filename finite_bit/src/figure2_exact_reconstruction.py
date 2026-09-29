"""Compute the Poisson and Klein–Gordon precision sweep."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from exact_oracle import reduced_exact
from pde_models import elliptic_blocks,kg_matrix,extract_blocks,schur_chain,cached_solve


def recursively_quantized_factors(A: np.ndarray, gamma: float, K: int, M: int=1):
    d,low,up=extract_blocks(A)
    Sref,Yref=schur_chain(d,low,up)
    S=[];Y=[];rows=[]
    for i, Di in enumerate(d):
        si=Di.copy() if i==0 else Di-low[i-1]@Y[-1]@up[i-1]
        yi=np.empty((2,2))
        for j in range(2):
            t=np.eye(2)[:,j]
            e=reduced_exact(si,t,gamma,M,K)
            y=np.asarray(e.decoded)
            yi[:,j]=y
            rows.append({'block':i+1,'column':j,'bitstring':e.bitstring,
                         'min_energy':e.residual_squared,
                         'column_residual':float(np.linalg.norm(si@y-t))})
        S.append(si);Y.append(yi)
    return S,Y,Sref,Yref,low,up,rows


def poisson_sweep_reference(nx:int=6,ny:int=2):
    """Discrete manufactured Poisson solution, zero trace on all four sides."""
    ix=np.arange(1,nx+1)/(nx+1)
    iy=np.arange(1,ny+1)/(ny+1)
    X,V=np.meshgrid(ix,iy,indexing='ij')
    return ((np.sin(np.pi*X)+.2*np.sin(2*np.pi*X))*np.sin(np.pi*V)).ravel()


def kg_sweep_trajectory(low,up,inverse,n:int=6,dt:float=.015,T:float=.3):
    """Unforced KG, c=1, mu=1.25; u=sin(pi*x), v=0 initially."""
    x=np.arange(1,n+1)/(n+1)
    u=np.sin(np.pi*x)
    v=np.zeros_like(u)
    steps=round(T/dt)
    assert abs(steps*dt-T)<1e-12
    for _ in range(steps):
        rhs=np.stack((u,v),axis=1).ravel()
        y=cached_solve(low,up,inverse,rhs).reshape(n,2)
        u,v=y.T
    return np.stack((u,v),axis=1).ravel()


def kg_sweep_continuum(n:int=6,T:float=.3,mu:float=1.25,c:float=1.):
    x=np.arange(1,n+1)/(n+1)
    wave=np.sin(np.pi*x)
    omega=np.sqrt(c*c*np.pi*np.pi+mu*mu)
    return np.stack((wave*np.cos(omega*T),-omega*wave*np.sin(omega*T)),axis=1).ravel()


def run_one(name:str,K:int):
    if K not in range(4,11):raise ValueError('Figure 2 uses K=4..10')
    if name=='poisson_2d':
        gamma=.01
        _,_,_,A,_,_,_=elliptic_blocks('poisson_2d',nx=6,ny=2)
    elif name=='klein_gordon_1d':
        gamma=1.
        A,*_=kg_matrix(n=6,dt=.015,c=1.,mu=1.25)
    else:raise ValueError(name)
    S,Y,Sref,Yref,low,up,records=recursively_quantized_factors(A,gamma,K)
    colres=np.array([r['column_residual'] for r in records])
    en=np.array([r['min_energy'] for r in records])
    inv_err=np.array([np.linalg.norm(v-vr,'fro') for v,vr in zip(Y,Yref)])
    inv_res=np.array([np.linalg.norm(s@v-np.eye(2),'fro') for s,v in zip(Sref,Y)])
    bounds=np.array([np.linalg.cond(s)*gamma*2**(-K) for s in Sref])
    r={'pde':name,'K':K,'B':2,'M':1,'gamma':gamma,'q_per_scalar':K+2,
       'qubo_bits':2*(K+2),'candidates_per_column':2**(2*(K+2)),
       'num_qubo_column_solves':len(records),'mean_qubo_energy':float(en.mean()),
       'max_qubo_energy':float(en.max()),'mean_column_residual':float(colres.mean()),
       'max_column_residual':float(colres.max()),
       'mean_inv_fro_error':float(inv_err.mean()),'max_inv_fro_error':float(inv_err.max()),
       'mean_inv_residual_fro':float(inv_res.mean()),'max_inv_residual_fro':float(inv_res.max()),
       'max_schur_condition':float(max(np.linalg.cond(s) for s in Sref)),
       'mean_theory_bound_fro':float(bounds.mean()),
       'max_theory_bound_fro':float(bounds.max()),
       'fixed_point_range_low':-2*gamma,
       'fixed_point_range_high':gamma*(2-2**(-K)),
       'exact_schur_inverse_min_entry':float(min(np.min(v) for v in Yref)),
       'exact_schur_inverse_max_entry':float(max(np.max(v) for v in Yref)),
       'clipping_risk':bool(any(np.any(v < -2*gamma) or np.any(v > gamma*(2-2**(-K))) for v in Yref)),
       'exact_block_vs_dense_rel_l2':0.}
    if name=='poisson_2d':
        ref=poisson_sweep_reference()
        b=A@ref
        exact=np.linalg.solve(A,b)
        quant=cached_solve(low,up,Y,b)
        defect=A@quant-b
        r.update(rel_l2_vs_discrete_exact=float(np.linalg.norm(quant-ref)/np.linalg.norm(ref)),
                 rel_l2_vs_dense=float(np.linalg.norm(quant-exact)/np.linalg.norm(exact)),
                 linf_vs_discrete_exact=float(np.max(np.abs(quant-ref))),
                 residual_norm=float(np.linalg.norm(defect)),
                 relative_residual=float(np.linalg.norm(defect)/np.linalg.norm(b)),
                 exact_block_vs_dense_rel_l2=float(np.linalg.norm(cached_solve(low,up,Yref,b)-exact)/np.linalg.norm(exact)))
    else:
        quant=kg_sweep_trajectory(low,up,Y)
        dense=kg_sweep_trajectory(low,up,Yref)
        reference=kg_sweep_continuum()
        r.update(num_time_steps=20,
                 rel_l2_vs_dense_final=float(np.linalg.norm(quant-dense)/np.linalg.norm(dense)),
                 rel_l2_vs_analytic_final=float(np.linalg.norm(quant-reference)/np.linalg.norm(reference)),
                 dense_rel_l2_vs_analytic_final=float(np.linalg.norm(dense-reference)/np.linalg.norm(reference)),
                 linf_vs_dense_final=float(np.max(np.abs(quant-dense))),
                 exact_block_vs_dense_rel_l2=float(np.linalg.norm(kg_sweep_trajectory(low,up,Yref)-dense)/np.linalg.norm(dense)))
    return r


def all_rows():
    return [run_one(n,k) for k in range(4,11) for n in ('poisson_2d','klein_gordon_1d')]
