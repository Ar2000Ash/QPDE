#!/usr/bin/env python3
"""Verify the 34 directly accessible reconstructed normalized QUBO input tables.

Independent of historical file formatting. Compares each of 300 published terms to
Q(S,gamma,j) derived from raw archived Schur blocks; checks full polynomial energy
on the two preserved reference/hardware states per QUBO. No QCI contact required.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'data'/'reconstructed_qubos'
RAW=ROOT/'data'/'raw'/'dirac3'/'reconstructed_60_instances.csv'

def weights(gamma:float)->np.ndarray:
    w=np.array([-2.,1.]+[2.**(-j) for j in range(1,11)],float)
    p=np.zeros((2,24));p[0,:12]=gamma*w;p[1,12:]=gamma*w
    return p

def run(input_dir:Path=INPUT, raw_file:Path=RAW)->None:
    refs=pd.read_csv(raw_file,dtype={'hw_best_bitstring':str,'global_min_bitstring':str})
    man=pd.read_csv(input_dir/'MANIFEST.csv',dtype=str)
    assert len(man)==34 and man.unique_id.nunique()==34 and man.representative_qci_id.nunique()==34
    stored=set(input_dir.glob('RECONSTRUCTED_*.csv'))
    assert len(stored)==34
    for row in man.itertuples(index=False):
        file=input_dir/row.filename
        assert file in stored and hashlib.sha256(file.read_bytes()).hexdigest()==row.reconstructed_file_sha256, row.unique_id
        rec=refs[refs.qci_id==row.representative_qci_id].iloc[0]
        assert rec.unique_id==row.unique_id
        df=pd.read_csv(file)
        assert len(df)==300 and df.bit_i_zero_based.min()==0 and df.bit_j_zero_based.max()==23
        assert (df.bit_i_zero_based<=df.bit_j_zero_based).all()
        pairs=list(zip(df.bit_i_zero_based,df.bit_j_zero_based))
        expected=[(i,j) for i in range(24) for j in range(i,24)]
        assert pairs==expected,(row.unique_id,'wrong order or index')
        S=np.array([[rec.S00,rec.S01],[rec.S10,rec.S11]],float)
        P=weights(float(rec.gamma));AP=S@P;G=AP.T@AP
        linear=-2*AP.T@np.eye(2)[:,int(rec.column)]
        Q=np.zeros((24,24))
        for i,j in expected:Q[i,j]=(G[i,i]+linear[i]) if i==j else 2*G[i,j]
        scale=max(abs(Q[np.triu_indices(24)]));assert abs(scale-rec.coeff_scale)<2e-11
        rows=np.array([Q[i,j] for i,j in pairs])
        assert np.max(abs(rows-df.unnormalized_qubo_coefficient.to_numpy()))<2e-11
        assert np.max(abs(rows/scale-df.normalized_qubo_coefficient.to_numpy()))<2e-11
        assert max(abs(df.normalized_qubo_coefficient))==1.
        for bits in (str(rec.global_min_bitstring).zfill(24),str(rec.hw_best_bitstring).zfill(24)):
            b=np.array([int(x) for x in bits],float)
            y=P@b;direct=(np.linalg.norm(S@y-np.eye(2)[:,int(rec.column)])**2-1)/scale
            polynomial=sum(v*b[i]*b[j] for (i,j),v in zip(pairs,df.normalized_qubo_coefficient))
            assert abs(direct-polynomial)<2e-12,(row.unique_id,'objective identity')
    print('34 directly accessible QUBO files PASSED: 300 coefficients each, reconstructed matrix and tested objectives.')

if __name__=='__main__':run()
