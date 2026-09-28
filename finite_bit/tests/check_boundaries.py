"""Check the discretized operators and boundary-condition fingerprints."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pde_models import (manufacture,forcing,elliptic_blocks,heat_matrix,kg_matrix,extract_blocks,schur_chain,integrate,cached_solve)

def close(label,a,b,atol=5e-12):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert err<atol,(label,err,a,b)
    return err

def run():
    n=20;x=np.arange(1,n+1)/21
    H,_,_,_=heat_matrix(n,nu=.08);cl=lambda A:np.linalg.cond(A)
    close('heat operator cond',cl(H),1.1394339187575457)
    close('heat manufactured first',manufacture('heat_1d',x[0],t=.05),.1870700283429833910)
    close('heat right boundary',manufacture('heat_1d',np.array(1.),t=.05),np.exp(-.05))
    _,low,up=extract_blocks(H);iv=schur_chain(*extract_blocks(H))[1]
    heat=integrate('heat_1d',iv,H)
    close('heat precise first',heat[0],.1870852273535272836)
    close('heat precise last',heat[-1],1.047733505898525097)
    B,_,_,_=heat_matrix(n,nu=.04)
    close('burgers operator cond',cl(B),1.0697444204176292)
    close('burgers manufactured first',manufacture('burgers_1d',x[0],t=.05),.3925862044131182560)
    close('burgers Dirichlet left',manufacture('burgers_1d',np.array(0.),t=.05),.35)
    close('burgers Dirichlet right',manufacture('burgers_1d',np.array(1.),t=.05),.5)
    burg=integrate('burgers_1d',schur_chain(*extract_blocks(B))[1],B)
    close('burgers precise first',burg[0],.3926235904673558985)
    close('burgers precise last',burg[-1],.5282406194822577561)
    KG,_,_,_=kg_matrix(20,.001,1.1,2.)
    close('Klein-Gordon operator cond',cl(KG),6.350787754621465)
    close('Klein-Gordon max Schur cond',max(np.linalg.cond(s) for s in schur_chain(*extract_blocks(KG))[0]),2.785090451255385)
    close('KG manufactured u first',manufacture('kg_u',x[0],t=.05),.1579936005538125354)
    close('KG manufactured v first',manufacture('kg_v',x[0],t=.05),-.02288521707660190821)
    k=integrate('klein_gordon_1d',schur_chain(*extract_blocks(KG))[1],KG)
    close('KG precise u first',k[0],.1579864602750547242)
    close('KG precise v first',k[1],-.02271548198497597842)
    for name,target_cond,target_first,trunc in [('poisson_2d',26.655602086111447,.3621693694796396246,10.753981592795974),('helmholtz_2d',50.21037880064371,.7543045062127848865,1.6048685801218667)]:
        d,l,u,A,exact,b,bc=elliptic_blocks(name)
        close(name+' cond',cl(A),target_cond)
        close(name+' manufactured first',exact[0],target_first)
        xx=np.array([(i/11,j/3) for i in range(1,11) for j in range(1,3)])
        close(name+' max continuous-discrete residual',np.max(np.abs(b-bc-forcing(name,xx[:,0],xx[:,1]))),trunc)
    print('PASS: five operators, manufactured fields, physical Dirichlet traces, and three complete precise time integrators match archive fingerprints')
if __name__=='__main__':run()
