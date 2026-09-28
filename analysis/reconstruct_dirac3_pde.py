#!/usr/bin/env python3
"""Independently rebuild QCI-derived cached Schur factors and terminal PDE solves.

Uses the preserved matrix entries, gamma and ORIGINAL HARDWARE BITSTRINGS, not
archived decoded hardware vectors or archived hardware solution columns.

The archived DENSE TERMINAL REFERENCE field defines a reproducible terminal RHS
b=A@u_ref. This verifies the entire cached linear solve and paper's terminal
Table 6 / Fig. 7, but does not recreate historical transient forcing/BC histories.
No QCI connection or original upload polynomial is required.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'dirac3'
OUT = ROOT / 'data' / 'processed'
LABELS = ('Heat 1D', 'Poisson 2D', 'Klein-Gordon 1D')
EXPECTED = {
    'Heat 1D': (0.00032391279008389326, 0.00033263717850195865),
    'Poisson 2D': (0.001970816149365653, 0.015508288121356788),
    'Klein-Gordon 1D': (0.00029703911071363637, 0.00029691333574971714),
}


def decode(bitstr: str, gamma: float) -> np.ndarray:
    if len(bitstr) != 24 or set(bitstr) - {'0','1'}:
        raise ValueError(f'Not an exact 24-bit bitstring: {bitstr!r}')
    out=[]
    for base in (0,12):
        b=bitstr[base:base+12]
        out.append(gamma*(-2*int(b[0]) + int(b[1]) + sum(int(b[j+1])*2.**(-j) for j in range(1,11))))
    return np.array(out,dtype=float)


def operator_blocks(name: str, schur: list[np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    D=schur[0].copy(); dS=D-schur[1]
    if name=='Heat 1D':
        # Adjacent 1D grid points couple across a two-point block boundary.
        r=-float(D[0,1]); E=np.array([[0.,-r],[0.,0.]]);U=E.T
        assert abs(r-0.03528)<1e-12
        meta={'coupling':r,'type':'one-dimensional tridiagonal'}
    elif name=='Poisson 2D':
        # Ten x-lines by two y-interior points; dx=1/11 and dy=1/3.
        inv=np.linalg.inv(D)
        c=float(np.sqrt(dS[0,0]/inv[0,0]))
        assert abs(c-121.)<1e-10
        E=U=-c*np.eye(2)
        meta={'x_laplacian_coupling':c,'y_laplacian_coupling':9.,'type':'two-point line blocks'}
    elif name=='Klein-Gordon 1D':
        # Interleaved (u_i,v_i) state. Only the v-equation couples u-neighbours.
        inv=np.linalg.inv(D)
        c=float(np.sqrt(dS[1,0]/inv[0,1]));assert abs(c-0.14641)<1e-11
        E=U=np.array([[0.,0.],[-c,0.]])
        meta={'offblock_wave_coupling':c,'type':'interleaved coupled first-order system'}
    else: raise ValueError(name)
    assert np.allclose(D-E@np.linalg.inv(D)@U,schur[1],rtol=0,atol=1e-11)
    return D,E,U,meta


def assemble(D:np.ndarray,E:np.ndarray,U:np.ndarray,nblocks:int)->np.ndarray:
    a=np.zeros((2*nblocks,2*nblocks))
    for i in range(nblocks):
        a[2*i:2*i+2,2*i:2*i+2]=D
        if i+1<nblocks:
            a[2*i+2:2*i+4,2*i:2*i+2]=E
            a[2*i:2*i+2,2*i+2:2*i+4]=U
    return a


def cached_solve(E:np.ndarray,U:np.ndarray,inv:list[np.ndarray],b:np.ndarray)->np.ndarray:
    n=len(inv); g=[None]*n;u=[None]*n
    g[0]=b[:2].copy()
    for i in range(1,n):g[i]=b[2*i:2*i+2]-E@inv[i-1]@g[i-1]
    u[-1]=inv[-1]@g[-1]
    for i in range(n-2,-1,-1):u[i]=inv[i]@(g[i]-U@u[i+1])
    return np.concatenate(u)


def get_ref_and_archived(raw:Path,name:str)->tuple[np.ndarray,np.ndarray,pd.DataFrame]:
    filename={'Heat 1D':'heat_solution.csv','Poisson 2D':'poisson_solution.csv','Klein-Gordon 1D':'klein_gordon_solution.csv'}[name]
    source=pd.read_csv(raw/filename)
    if name!='Klein-Gordon 1D':
        ref=source.dense_ground_truth.to_numpy(float)
        archival=source.qci_quantum.to_numpy(float)
    else:
        ref=np.column_stack((source.u_dense_ground_truth,source.v_dense_ground_truth)).ravel()
        archival=np.column_stack((source.u_qci_quantum,source.v_qci_quantum)).ravel()
    assert len(ref)==len(archival)==20
    return ref,archival,source


def regenerate(raw:Path=RAW, out:Path=OUT, write:bool=False, strict:bool=True)->dict:
    rows=pd.read_csv(raw/'reconstructed_60_instances.csv',dtype={'hw_best_bitstring':str,'global_min_bitstring':str,'hw_exact_bitstring':str})
    compact=pd.read_csv(raw/'mapped_60_instances.csv',dtype={'dirac_bitstring':str,'exact_bitstring':str})
    assert len(rows)==len(compact)==60
    rows=rows.set_index('qci_id'); compact=compact.set_index('qci_id')
    all_fields={};table=[];detail=[]
    for name in LABELS:
        source=rows[rows.pde_name==name]
        assert len(source)==20
        schur=[];hw_inv=[];exact_inv=[]
        for bi in range(1,11):
            both=source[source.block==bi].sort_values('column')
            assert list(both.column)==[0,1]
            S=np.array([[both.iloc[0].S00,both.iloc[0].S01],[both.iloc[0].S10,both.iloc[0].S11]],float)
            schur.append(S)
            hv=np.empty((2,2));ev=np.empty((2,2))
            for col,(id_,rec) in enumerate(both.iterrows()):
                mapped=compact.loc[id_]
                assert rec.unique_id==mapped.unique_id and rec.column==col and rec.block==mapped.block
                bits=str(rec.hw_best_bitstring); assert bits==str(mapped.dirac_bitstring)
                exact_bits=str(rec.global_min_bitstring)
                assert exact_bits==str(mapped.exact_bitstring)
                y=decode(bits,float(rec.gamma));ey=decode(exact_bits,float(rec.gamma));hv[:,col]=y;ev[:,col]=ey
                target=np.eye(2)[:,col];r=np.linalg.norm(S@y-target);er=np.linalg.norm(S@ey-target)
                assert np.isclose(r,rec.hw_best_inverse_residual,atol=1e-10,rtol=1e-9)
                assert np.isclose(r,mapped.local_inverse_residual_dirac,atol=1e-10,rtol=1e-9)
                assert np.isclose(er,mapped.local_inverse_residual_exact,atol=1e-10,rtol=1e-9)
                energy=(r*r-1)/rec.coeff_scale
                assert np.isclose(energy,mapped.dirac_objective,atol=1e-10)
                detail.append({'qci_id':id_,'pde_name':name,'block':bi,'column':col,'unique_id':rec.unique_id,'hw_bitstring':bits,'local_residual':r,'exact_residual':er,'normalized_objective':energy})
            hw_inv.append(hv);exact_inv.append(ev)
        D,E,U,meta=operator_blocks(name,schur)
        actual=[D.copy()]
        for i in range(1,10):actual.append(D-E@np.linalg.inv(actual[-1])@U)
        largest_schur=max(float(np.max(np.abs(a-b))) for a,b in zip(actual,schur))
        assert largest_schur<1e-10,(name,largest_schur)
        A=assemble(D,E,U,10)
        reference,archived,orig=get_ref_and_archived(raw,name)
        # Precisely defined benchmark terminal RHS. Does NOT claim recovered time histories.
        rhs=A@reference
        computed=cached_solve(E,U,hw_inv,rhs)
        exact_comp=cached_solve(E,U,exact_inv,rhs)
        classical=cached_solve(E,U,[np.linalg.inv(s) for s in actual],rhs)
        err=float(np.linalg.norm(computed-reference)/np.linalg.norm(reference))
        resid=float(np.linalg.norm(A@computed-rhs)/np.linalg.norm(rhs))
        delta=float(np.max(np.abs(computed-archived)))
        assert delta<2e-12,(name,'field mismatch',delta)
        assert np.max(np.abs(classical-reference))<2e-12
        if strict:
            assert np.allclose((err,resid),EXPECTED[name],atol=5e-11,rtol=0),(name,err,resid)
        summary={'pde':name,'relative_error':err,'relative_residual':resid,'max_abs_error':float(np.max(np.abs(computed-reference))),
                 'max_abs_difference_from_archived_hardware_field':delta,'max_abs_schur_reconstruction_difference':largest_schur,
                 'exact_reference_bitstring_solution_relative_error':float(np.linalg.norm(exact_comp-reference)/np.linalg.norm(reference)),
                 'rhs_construction':'A @ archived dense terminal reference (not archived transient forcing history)'}
        table.append(summary)
        if name!='Klein-Gordon 1D':
            frame=pd.DataFrame({'index':orig['index'].astype(int),'dense_ground_truth':reference,'qci_quantum':computed,'abs_error':abs(computed-reference)})
            output='dirac_heat.csv' if name=='Heat 1D' else 'dirac_poisson.csv'
        else:
            frame=pd.DataFrame({'space_index':orig.space_index.astype(int),'u_dense_ground_truth':reference[::2],
                  'u_qci_quantum':computed[::2],'u_abs_error':abs(computed[::2]-reference[::2]),
                  'v_dense_ground_truth':reference[1::2],'v_qci_quantum':computed[1::2],
                  'v_abs_error':abs(computed[1::2]-reference[1::2])})
            output='dirac_kg.csv'
        all_fields[output]=frame
        if name=='Poisson 2D':
            slices=[]
            for slice_id,offset in ((1,0),(2,1)):
                for xi,i in enumerate(range(offset,20,2),start=1):
                    slices.append({'x_index':xi,'slice':slice_id,'dense':reference[i],'hardware':computed[i],
                                  'abs_error':abs(computed[i]-reference[i])})
            all_fields['dirac_poisson_slices.csv']=pd.DataFrame(slices)
        # Additional fully explicit terminal linear system: 20x20 A, b, reference, hardware.
        all_fields[f'dirac_{name.lower().replace(" ","_").replace("-","_")}_terminal_rhs.csv']=pd.DataFrame({'index':np.arange(1,21),'rhs':rhs,'dense_reference':reference,'hardware_reconstructed':computed,'exact_reference_bitstrings':exact_comp})
    full=pd.DataFrame(table); all_fields['dirac_pde_independent_reconstruction.csv']=full
    all_fields['dirac_local_inverse_audit.csv']=pd.DataFrame(detail)
    if write:
        out.mkdir(parents=True,exist_ok=True)
        for filename,table in all_fields.items():table.to_csv(out/filename,index=False,float_format='%.17g')
    return {'summary':full,'fields':all_fields}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=RAW)
    parser.add_argument('--out-dir',type=Path,default=OUT)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    result=regenerate(args.raw_dir,args.out_dir,args.write)
    print(result['summary'].to_string(index=False))
    print('All three archived terminal fields reconstructed from saved bitstrings and Schur blocks.')
