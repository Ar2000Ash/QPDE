"""CPU independent full-grid reference versus analytically reduced exact oracle."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exact_oracle import reduced_exact,decode_bits,canonical_bits

def exhaustive_cpu(S,target,gamma,M=1,K=4):
    bound=2**(M+K); codes=np.arange(-bound,bound,dtype=float); scale=gamma/2**K
    x,y=np.meshgrid(codes,codes,indexing='ij')
    rr=(S[:,0,None,None]*(scale*x)[None,:,:]+S[:,1,None,None]*(scale*y)[None,:,:]-target[:,None,None])
    energies=np.sum(rr*rr,axis=0)
    idx=np.unravel_index(np.argmin(energies),energies.shape)
    return (int(codes[idx[0]]),int(codes[idx[1]]),float(energies[idx]))

def run():
    rng=np.random.default_rng(20260927)
    for k in [2,3,4,5,6]:
        for case in range(50):
            S=rng.normal(size=(2,2))+2*np.eye(2)
            target=rng.normal(size=2)
            gamma=float(rng.uniform(.05,2.))
            result=reduced_exact(S,target,gamma,K=k)
            i,j,e=exhaustive_cpu(S,target,gamma,K=k)
            assert abs(result.residual_squared-e)<5e-12, (k,case,result.residual_squared,e)
            if abs(result.residual_squared-e)>1e-13: print('near tie')
    print('PASS: reduced oracle agrees with independently brute-forced CPU objectives for 250 random matrices across K=2..6')
if __name__=='__main__':run()
