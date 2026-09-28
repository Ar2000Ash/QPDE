"""Single-grid residual-correction control for the five-PDE experiment."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
import numpy as np
from exact_oracle import reduced_exact
from operators import fixed_operator
from pde_models import manufacture, forcing, extract_blocks, schur_chain, cached_solve, integrate

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('heat_1d','burgers_1d','poisson_2d','helmholtz_2d','klein_gordon_1d')
GAMMA = {'heat_1d':.60,'burgers_1d':.60,'poisson_2d':.01,
         'helmholtz_2d':.01,'klein_gordon_1d':1.00}
M=1
K=10
PASSES=3
B=2
DT=.001
T=.05


def factorize_fixed(name: str):
    """Keep gamma unchanged for all blocks/columns/stages; QUBO for EACH step."""
    A, elliptic = fixed_operator(name)
    diag, lower, upper = extract_blocks(A,B)
    Schur=[]; inverses=[]; record=[]
    start=time.perf_counter()
    for i, Di in enumerate(diag):
        S=Di.copy() if not i else Di - lower[i-1]@inverses[-1]@upper[i-1]
        Yi=np.empty((B,B))
        for j in range(B):
            e=np.eye(B)[:,j]
            y=np.zeros(B)
            for stage in range(1+PASSES):
                rhs=e-S@y
                candidate=reduced_exact(S,rhs,GAMMA[name],M=M,K=K)
                delta=np.asarray(candidate.decoded)
                y=y+delta
                record.append(dict(pde=name,block=i+1,column=j,stage=stage,
                    gamma=GAMMA[name],bitstring=candidate.bitstring,
                    delta0=float(delta[0]),delta1=float(delta[1]),
                    energy=float(candidate.residual_squared),
                    post_column_residual=float(np.linalg.norm(S@y-e))))
            Yi[:,j]=y
        Schur.append(S);inverses.append(Yi)
    elapsed=time.perf_counter()-start
    return A,elliptic,lower,upper,Schur,inverses,record,elapsed


def integrate_with_last_rhs(name, inverses, A):
    """Time integrators and Dirichlet traces matching pde_models.integrate.

    Returns final RHS computed from the approximate trajectory to define the
    reported final *algebraic* residual, which cannot be A@x - b for time cases
    using an unrelated manufactured endpoint.
    """
    grid=np.arange(1,21)/21
    _,lower,upper=extract_blocks(A,B)
    rhs=None
    if name in ('heat_1d','burgers_1d'):
        r=(.08 if name=='heat_1d' else .04)*DT*21**2
        x=manufacture(name,grid,t=0.)
        for k in range(1,round(T/DT)+1):
            t=k*DT;rhs=x.copy()
            if name=='burgers_1d':
                old=np.r_[manufacture(name,np.array(0.),t=t-DT),x,manufacture(name,np.array(1.),t=t-DT)]
                rhs-=DT*x*(old[2:]-old[:-2])/(2/21)
                rhs+=DT*forcing(name,grid,t=t-DT)
            else:rhs+=DT*forcing(name,grid,t=t)
            rhs[0]+=r*manufacture(name,np.array(0.),t=t)
            rhs[-1]+=r*manufacture(name,np.array(1.),t=t)
            x=cached_solve(lower,upper,inverses,rhs)
        return x,rhs
    if name=='klein_gordon_1d':
        u=manufacture('kg_u',grid,t=0.);v=manufacture('kg_v',grid,t=0.)
        for k in range(1,round(T/DT)+1):
            t=k*DT
            rhs=np.stack((u,v+DT*forcing('kg_v',grid,t=t,c=1.1,mu=2.)),axis=1).ravel()
            rhs[-1]+=DT*1.1**2*21**2*manufacture('kg_u',np.array(1.),t=t)
            state=cached_solve(lower,upper,inverses,rhs).reshape(20,B)
            u,v=state.T
        return np.stack((u,v),axis=1).ravel(),rhs
    raise ValueError('Only time-dependent PDEs use integrate_with_last_rhs')


def compute(name):
    if name not in NAMES:raise ValueError(name)
    A,ell,lower,upper,ss,Y,logs,pretime=factorize_fixed(name)
    d,l,u=extract_blocks(A,B)
    _,dense_factor=schur_chain(d,l,u)
    if ell is None:
        result,rhs=integrate_with_last_rhs(name,Y,A)
        reference=integrate(name,dense_factor,A)
        # Reuse the separately coded standard integrator as an independent check.
        assert np.max(np.abs(result-integrate(name,Y,A)))<2e-13
        grid=np.arange(1,21)/21
        if name=='klein_gordon_1d':
            exact=np.stack((manufacture('kg_u',grid,t=T),manufacture('kg_v',grid,t=T)),axis=1).ravel()
        else:exact=manufacture(name,grid,t=T)
    else:
        exact,rhs,_=ell
        result=cached_solve(lower,upper,Y,rhs)
        reference=np.linalg.solve(A,rhs)
    rho=[np.linalg.norm(S@Yi-np.eye(B),ord='fro') for S,Yi in zip(ss,Y)]
    first=[x for x in logs if x['stage']==0]
    refinements=[x for x in logs if x['stage']>0]
    max_ref_step=max(max(abs(x['delta0']),abs(x['delta1'])) for x in refinements)
    rel=np.linalg.norm(result-reference)/np.linalg.norm(reference)
    corr=float(np.corrcoef(result,reference)[0,1])
    return dict(pde=name,label={'heat_1d':'Heat 1D','burgers_1d':'Burgers 1D','poisson_2d':'Poisson 2D',
                 'helmholtz_2d':'Helmholtz 2D','klein_gordon_1d':'Klein--Gordon 1D'}[name],
       protocol='SINGLE_GRID_CONTROL', gamma=GAMMA[name], B=B,M=M,K=K,
       correction_passes=PASSES, scale_policy='same_gamma_each_stage',
       schur_policy='quantized_recursive',N=A.shape[0],cond_A=float(np.linalg.cond(A)),
       rel_dense=float(rel),rel_exact=float(np.linalg.norm(result-exact)/np.linalg.norm(exact)),
       linf_exact=float(np.max(abs(result-exact))),
       r_inf=float(np.max(abs(A@result-rhs))),
       relative_residual=float(np.linalg.norm(A@result-rhs)/np.linalg.norm(rhs)),
       max_inv_res=float(max(rho)),mean_inv_res=float(np.mean(rho)),
       max_schur_cond=float(max(np.linalg.cond(S) for S in ss)),
       calls=len(logs),search_space_states_per_call=2**(B*(1+M+K)),
       theoretical_candidate_product=len(logs)*2**(B*(1+M+K)),
       precompute_cpu_reduced_seconds=pretime,
       final_iteration_nonzero_corrections=sum((x['delta0']!=0 or x['delta1']!=0) for x in refinements),
       max_correction_magnitude=max_ref_step,
       mean_qubo_energy=float(np.mean([x['energy'] for x in logs])),
       max_qubo_energy=float(max(x['energy'] for x in logs)),
       correlation=corr),result,reference,logs,ss,Y


def generate(out=ROOT/'outputs'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];events=[];trajectory={}
    for name in NAMES:
        row,x,ref,log,ss,Y=compute(name);rows.append(row);events.extend(log)
        for i,(v,z) in enumerate(zip(x,ref)):
            trajectory[(name,i)]={'pde':name,'component_index':i,'single_grid':float(v),'dense_numerical':float(z)}
    with (out/'figure3_control.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    with (out/'figure3_control_stage_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,events[0].keys());w.writeheader();w.writerows(events)
    with (out/'figure3_control_fields.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,next(iter(trajectory.values())).keys());w.writeheader();w.writerows(trajectory.values())
    (out/'figure3_control_metadata.json').write_text(json.dumps(dict(
        protocol='SINGLE_GRID_CONTROL', backend='exact reduced-grid CPU',
        gamma=GAMMA, correction_passes=PASSES, table_rows=len(rows),
        stage_rows=len(events)), indent=2)+'\n')
    for r in rows:print(r['pde'], 'rel_dense',format(r['rel_dense'],'.12g'),
                         'r_inf',format(r['r_inf'],'.12g'),
                         'max_inv_res',format(r['max_inv_res'],'.12g'),
                         'refinement_updates',r['final_iteration_nonzero_corrections'],
                         'calls',r['calls'])
    return rows

if __name__=='__main__':generate()
