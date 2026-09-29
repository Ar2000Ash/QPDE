"""Compute five-PDE results with four prescribed correction scales."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from exact_oracle import reduced_exact, decode_bits
from operators import fixed_operator
from figure3_control import (NAMES, GAMMA, M, K, PASSES, B, DT, T,
                                        integrate_with_last_rhs)
from pde_models import manufacture, extract_blocks, schur_chain, cached_solve, integrate

ROOT=Path(__file__).resolve().parents[1]
SHRINK=8
SCHEME='GEOMETRIC_MULTISCALE'


def stage_gamma(name, stage):
    if name not in NAMES or stage not in range(PASSES+1):
        raise ValueError((name,stage))
    return GAMMA[name]/SHRINK**stage


def factorize(name):
    A,ell=fixed_operator(name)
    diag,lower,upper=extract_blocks(A,B)
    blocks=[];inverses=[];events=[]
    started=time.perf_counter()
    for i, Di in enumerate(diag):
        S=Di.copy() if i==0 else Di-lower[i-1]@inverses[-1]@upper[i-1]
        Y=np.empty((B,B))
        for j in range(B):
            e=np.eye(B)[:,j]
            y=np.zeros(B)
            for stage in range(PASSES+1):
                r=e-S@y
                gamma=stage_gamma(name,stage)
                audit_correction=np.linalg.solve(S,r)
                high=gamma*(2**M-2**(-K));low=-gamma*2**M
                clipped=bool(np.any(audit_correction<low) or np.any(audit_correction>high))
                z=reduced_exact(S,r,gamma,M=M,K=K)
                delta=np.asarray(z.decoded)
                assert np.array_equal(delta,decode_bits(z.bitstring,gamma,M,K))
                y=y+delta
                events.append(dict(pde=name,block=i+1,column=j,stage=stage,
                                   gamma=gamma,bitstring=z.bitstring,
                                   delta0=float(delta[0]),delta1=float(delta[1]),
                                   energy=float(z.residual_squared),
                                   post_column_residual=float(np.linalg.norm(e-S@y)),
                                   continuous_correction_within_grid=not clipped,
                                   max_abs_continuous_correction=float(np.max(np.abs(audit_correction)))))
            Y[:,j]=y
        blocks.append(S);inverses.append(Y)
    elapsed=time.perf_counter()-started
    return A,ell,lower,upper,blocks,inverses,events,elapsed


def manufactured_reference(name,ell):
    if ell is not None:return ell[0]
    x=np.arange(1,21)/21
    if name=='klein_gordon_1d':
        return np.stack((manufacture('kg_u',x,t=T),manufacture('kg_v',x,t=T)),axis=1).ravel()
    return manufacture(name,x,t=T)


def compute(name):
    if name not in NAMES:raise ValueError(name)
    A,ell,lower,upper,blocks,Y,events,elapsed=factorize(name)
    _,Yref=schur_chain(*extract_blocks(A,B))
    if ell is None:
        x,b=integrate_with_last_rhs(name,Y,A)
        dense=integrate(name,Yref,A)
        assert np.max(abs(x-integrate(name,Y,A)))<2e-13
    else:
        _,b,_=ell
        x=cached_solve(lower,upper,Y,b)
        dense=np.linalg.solve(A,b)
    manufactured=manufactured_reference(name,ell)
    dif=x-dense
    residuals=[float(np.linalg.norm(S@inv-np.eye(B),ord='fro')) for S,inv in zip(blocks,Y)]
    corrections=[e for e in events if e['stage']>0]
    clipped=[e for e in events if not e['continuous_correction_within_grid']]
    nonzero=[sum((e['delta0']!=0 or e['delta1']!=0) for e in events if e['stage']==p) for p in range(PASSES+1)]
    assert len(events)==A.shape[0]*(PASSES+1)
    row=dict(pde=name,label={
          'heat_1d':'Heat 1D','burgers_1d':'Burgers 1D',
          'poisson_2d':'Poisson 2D','helmholtz_2d':'Helmholtz 2D',
          'klein_gordon_1d':'Klein--Gordon 1D'}[name],
        protocol=SCHEME,
        gamma0=GAMMA[name],shrink=SHRINK,
        gamma0_stage=stage_gamma(name,0),gamma1=stage_gamma(name,1),
        gamma2=stage_gamma(name,2),gamma3=stage_gamma(name,3),
        B=B,M=M,K=K,correction_passes=PASSES,
        scale_policy='gamma_p=gamma0/8**p',schur_policy='quantized_recursive',
        N=A.shape[0],cond_A=float(np.linalg.cond(A)),
        rel_dense=float(np.linalg.norm(dif)/np.linalg.norm(dense)),
        rel_exact=float(np.linalg.norm(x-manufactured)/np.linalg.norm(manufactured)),
        dense_rel_exact=float(np.linalg.norm(dense-manufactured)/np.linalg.norm(manufactured)),
        linf_exact=float(np.max(np.abs(x-manufactured))),
        r_inf=float(np.max(np.abs(A@x-b))),
        relative_residual=float(np.linalg.norm(A@x-b)/np.linalg.norm(b)),
        max_inv_res=float(max(residuals)),mean_inv_res=float(np.mean(residuals)),
        max_schur_cond=float(max(np.linalg.cond(S) for S in blocks)),
        calls=len(events),logical_bits_per_call=B*(1+M+K),
        logical_states_per_call=2**(B*(1+M+K)),
        logical_state_space_accounting=len(events)*2**(B*(1+M+K)),
        precompute_cpu_reduced_seconds=elapsed,
        nonzero_correction_updates=sum(e['delta0']!=0 or e['delta1']!=0 for e in corrections),
        clipped_continuous_corrections=len(clipped),
        stage0_nonzero=nonzero[0],stage1_nonzero=nonzero[1],
        stage2_nonzero=nonzero[2],stage3_nonzero=nonzero[3],
        max_qubo_energy=float(max(e['energy'] for e in events)),
        mean_qubo_energy=float(np.mean([e['energy'] for e in events])),
        correlation=float(np.corrcoef(x,dense)[0,1]))
    if name=='klein_gordon_1d':
        for label,k in (('u',0),('v',1)):
            xx=x[k::2];rr=dense[k::2];ee=manufactured[k::2]
            row[f'{label}_rel_dense']=float(np.linalg.norm(xx-rr)/np.linalg.norm(rr))
            row[f'{label}_rel_exact']=float(np.linalg.norm(xx-ee)/np.linalg.norm(ee))
            row[f'{label}_max_abs_error']=float(np.max(abs(xx-rr)))
    else:
        for label in ('u','v'):
            row[f'{label}_rel_dense']=row[f'{label}_rel_exact']=row[f'{label}_max_abs_error']=''
    return row,x,dense,manufactured,events,blocks,Y


def generate(out=None):
    out=Path(out) if out is not None else ROOT/'outputs'
    out.mkdir(parents=True,exist_ok=True)
    rows=[];events=[];fields=[]
    for name in NAMES:
        r,x,dense,exact,stage,_,_=compute(name)
        rows.append(r);events.extend(stage)
        fields.extend(dict(pde=name,component_index=i,multiscale=float(v),
                           dense_numerical=float(d),manufactured=float(a))
                      for i,(v,d,a) in enumerate(zip(x,dense,exact)))
        print(f'{name}: e_dense={r["rel_dense"]:.12g} rho_S={r["max_inv_res"]:.12g} '
              f'updates={r["nonzero_correction_updates"]}, clipping={r["clipped_continuous_corrections"]}, calls={r["calls"]}')
    for filename,vals in [('figure3_multiscale.csv',rows),('figure3_multiscale_stage_log.csv',events),
                          ('figure3_multiscale_fields.csv',fields)]:
        with (out/filename).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,vals[0].keys());w.writeheader();w.writerows(vals)
    (out/'figure3_multiscale_metadata.json').write_text(json.dumps({
        'protocol': SCHEME, 'backend': 'exact reduced-grid CPU',
        'gamma0': GAMMA, 'shrink': SHRINK, 'stages': PASSES+1,
        'hyperparameters': {'B': B, 'M': M, 'K': K, 'dt': DT, 'T': T},
        'schur': 'recursive quantized inverse',
        'total_calls': len(events),
        'total_correction_steps': sum(e['stage']>0 for e in events),
        'clipped_audit_corrections': sum(not e['continuous_correction_within_grid'] for e in events),
    }, indent=2)+'\n', encoding='utf8')
    return rows

if __name__=='__main__':generate()
