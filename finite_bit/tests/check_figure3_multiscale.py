"""Check correction scales, inverse columns, and PDE results."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from figure3_multiscale import compute,stage_gamma,NAMES,GAMMA,PASSES,B,M,K,SHRINK
from figure3_control import compute as fixed_compute
from exact_oracle import decode_bits

EXPECTED={
 'heat_1d':(1.3367080088581643e-5,8.341045249140769e-7),
 'burgers_1d':(1.0986839891150165e-5,7.141050527137744e-7),
 'poisson_2d':(3.0522183023661557e-6,3.4461308764777923e-6),
 'helmholtz_2d':(6.884600474089066e-6,3.826884827787936e-6),
 'klein_gordon_1d':(2.3027081329091724e-5,8.610285564378199e-7),
}

def verify():
    total=0; updates=0; clipped=0
    for name in NAMES:
        row,x,dense,manufactured,events,Slist,Ylist=compute(name)
        baseline,*_=fixed_compute(name)
        assert row['N']==(40 if name=='klein_gordon_1d' else 20)
        assert (row['B'],row['M'],row['K'],row['correction_passes'],row['shrink'])==(2,1,10,3,8)
        assert len(events)==row['N']*(PASSES+1)==row['calls']
        assert row['logical_bits_per_call']==24 and row['logical_states_per_call']==2**24
        assert row['rel_dense']<float(baseline['rel_dense'])
        assert row['max_inv_res']<float(baseline['max_inv_res'])
        assert row['nonzero_correction_updates']==3*row['N']
        assert row['clipped_continuous_corrections']==0
        np.testing.assert_allclose((row['rel_dense'],row['max_inv_res']),EXPECTED[name],atol=3e-12,rtol=1e-8)
        assert np.linalg.norm(x-dense)/np.linalg.norm(dense)==row['rel_dense']
        assert max(np.linalg.norm(S@Y-np.eye(B),'fro') for S,Y in zip(Slist,Ylist))==row['max_inv_res']
        by={(int(e['block']),int(e['column']),int(e['stage'])):e for e in events}
        for i,S in enumerate(Slist):
            for j in range(B):
                y=np.zeros(B);e=np.eye(B)[:,j]
                for p in range(PASSES+1):
                    z=by[(i+1,j,p)]
                    g=stage_gamma(name,p)
                    assert z['gamma']==g==GAMMA[name]/8**p
                    bits=z['bitstring'];assert len(bits)==24 and set(bits)<={'0','1'}
                    delta=decode_bits(bits,g,M,K)
                    np.testing.assert_allclose(delta,[z['delta0'],z['delta1']],rtol=0,atol=1e-17)
                    pre=e-S@y
                    np.testing.assert_allclose(np.linalg.norm(S@delta-pre)**2,z['energy'],atol=3e-14,rtol=3e-10)
                    y+=delta
                    np.testing.assert_allclose(np.linalg.norm(e-S@y),z['post_column_residual'],atol=3e-13,rtol=1e-9)
                    if p>0:assert np.any(delta!=0),(name,i,j,p)
                    assert z['continuous_correction_within_grid'],(name,i,j,p)
                np.testing.assert_allclose(y,Ylist[i][:,j],atol=2e-13,rtol=0)
        if name=='klein_gordon_1d':
            eu=np.linalg.norm((x-dense)[::2])/np.linalg.norm(dense[::2]);ev=np.linalg.norm((x-dense)[1::2])/np.linalg.norm(dense[1::2])
            np.testing.assert_allclose((eu,ev),[row['u_rel_dense'],row['v_rel_dense']],atol=1e-15)
            assert ev<1.3e-4
        total+=len(events);updates+=row['nonzero_correction_updates'];clipped+=row['clipped_continuous_corrections']
    assert(total,updates,clipped)==(480,360,0)
    print('PASS: all five PDEs, 480 exact finite-bit calls, 360 nonzero corrections, zero out-of-range audit corrections, unchanged B/M/K and boundaries')

if __name__=='__main__':verify()
