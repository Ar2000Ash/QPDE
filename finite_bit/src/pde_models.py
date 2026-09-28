"""Finite-difference operators, manufactured solutions, and boundary traces."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def manufacture(name: str,x:np.ndarray,y:np.ndarray|None=None,t:float=0.) -> np.ndarray:
    if name=='heat_1d':return np.exp(-t)*(np.sin(np.pi*x)+x)
    if name=='burgers_1d':return .35+.15*x+.25*np.exp(-t)*np.sin(np.pi*x)
    if name=='kg_u':return np.cos(1.7*t)*(np.sin(np.pi*x)+.2*x)
    if name=='kg_v':return -1.7*np.sin(1.7*t)*(np.sin(np.pi*x)+.2*x)
    if y is None:raise ValueError('2D case requires y')
    if name=='poisson_2d':return np.sin(np.pi*x)*np.sin(2*np.pi*y)+.15*x+.3*y+.15*x*y
    if name=='helmholtz_2d':return np.cos(2*np.pi*x)*np.sin(np.pi*y)+.1*x+.05*y
    raise KeyError(name)


def forcing(name:str,x:np.ndarray,y:np.ndarray|None=None,t:float=0.,nu:float|None=None,c:float=1.1,mu:float=2.)->np.ndarray:
    if name=='heat_1d':
        nu=.08 if nu is None else nu
        return np.exp(-t)*(-(np.sin(np.pi*x)+x)+nu*np.pi**2*np.sin(np.pi*x))
    if name=='burgers_1d':
        nu=.04 if nu is None else nu;e=np.exp(-t);u=manufacture(name,x,t=t)
        return -.25*e*np.sin(np.pi*x)+u*(.15+.25*e*np.pi*np.cos(np.pi*x))+nu*.25*e*np.pi**2*np.sin(np.pi*x)
    if name=='kg_v':
        phi=np.sin(np.pi*x)+.2*x
        return np.cos(1.7*t)*((mu*mu-1.7**2)*phi+c*c*np.pi**2*np.sin(np.pi*x))
    if name=='poisson_2d':return 5*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y)
    if name=='helmholtz_2d':return (5*np.pi**2-9)*np.cos(2*np.pi*x)*np.sin(np.pi*y)-9*(.1*x+.05*y)
    raise KeyError(name)


def elliptic_blocks(name:str,nx:int=10,ny:int=2)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if name not in ('poisson_2d','helmholtz_2d'):raise KeyError(name)
    hx=1/(nx+1);hy=1/(ny+1);ax=1/hx**2;ay=1/hy**2
    diag=2*(ax+ay)-(9 if name=='helmholtz_2d' else 0)
    D=np.diag([diag]*ny)+np.diag([-ay]*(ny-1),1)+np.diag([-ay]*(ny-1),-1)
    L=U=-ax*np.eye(ny)
    A=assemble_blocks([D]*nx,[L]*(nx-1),[U]*(nx-1))
    xy=np.array([(i*hx,j*hy) for i in range(1,nx+1) for j in range(1,ny+1)])
    u=manufacture(name,xy[:,0],xy[:,1]);b=A@u
    boundary=np.zeros(nx*ny)
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            row=(i-1)*ny+j-1
            for di,dj,coupling in [(1,0,ax),(-1,0,ax),(0,1,ay),(0,-1,ay)]:
                ni,nj=i+di,j+dj
                if not (1<=ni<=nx and 1<=nj<=ny):
                    boundary[row]+=coupling*manufacture(name,np.array(ni*hx),np.array(nj*hy))
    # b = (continuous source sampled on grid + boundary) plus a discretization
    # correction. For paper's discrete-manufactured case, b=A@u exactly.
    return D,L,U,A,u,b,boundary


def heat_matrix(n:int=20,dt:float=.001,nu:float=.08)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    r=nu*dt*(n+1)**2
    D=np.eye(2)*(1+2*r);L=U=-r*np.eye(2)
    A=assemble_blocks([D]*(n//2),[L]*(n//2-1),[U]*(n//2-1))
    # Physical one-dimensional tridiagonal ordering uses neighbor -r at both
    # odd/even intra-block positions, unlike the simple block U=-r I above.
    A=np.diag([1+2*r]*n)+np.diag([-r]*(n-1),1)+np.diag([-r]*(n-1),-1)
    return A, A[:2,:2].copy(), A[2:4,:2].copy(), A[:2,2:4].copy()


def kg_matrix(n:int=20,dt:float=.001,c:float=1.1,mu:float=2.)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    h=1/(n+1);a=dt*c*c/h**2
    D=np.array([[1.,-dt],[2*a+dt*mu*mu,1.]])
    L=U=np.array([[0.,0.],[-a,0.]])
    A=assemble_blocks([D]*n,[L]*(n-1),[U]*(n-1))
    return A,D,L,U


def assemble_blocks(diag:list[np.ndarray],low:list[np.ndarray],up:list[np.ndarray])->np.ndarray:
    n=len(diag);size=diag[0].shape[0];A=np.zeros((n*size,n*size))
    for i in range(n):
        A[i*size:(i+1)*size,i*size:(i+1)*size]=diag[i]
        if i<n-1:A[i*size:(i+1)*size,(i+1)*size:(i+2)*size]=up[i]
        if i>0:A[i*size:(i+1)*size,(i-1)*size:i*size]=low[i-1]
    return A


def extract_blocks(A:np.ndarray,B:int=2)->tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray]]:
    n=A.shape[0]//B
    d=[A[i*B:(i+1)*B,i*B:(i+1)*B].copy() for i in range(n)]
    low=[A[(i+1)*B:(i+2)*B,i*B:(i+1)*B].copy() for i in range(n-1)]
    up=[A[i*B:(i+1)*B,(i+1)*B:(i+2)*B].copy() for i in range(n-1)]
    return d,low,up


def schur_chain(d:list[np.ndarray],low:list[np.ndarray],up:list[np.ndarray]):
    ss=[];vv=[]
    for i,D in enumerate(d):
        S=D.copy() if i==0 else D-low[i-1]@vv[-1]@up[i-1]
        ss.append(S);vv.append(np.linalg.inv(S))
    return ss,vv


def cached_solve(low:list[np.ndarray],up:list[np.ndarray],inverse:list[np.ndarray],rhs:np.ndarray)->np.ndarray:
    n=len(inverse);B=inverse[0].shape[0];g=[]
    for i in range(n):
        v=rhs[i*B:(i+1)*B].copy()
        if i:v-=low[i-1]@inverse[i-1]@g[-1]
        g.append(v)
    x=[None]*n
    for i in range(n-1,-1,-1):
        v=g[i].copy()
        if i<n-1:v-=up[i]@x[i+1]
        x[i]=inverse[i]@v
    return np.concatenate(x)


def integrate(name:str,inverse:list[np.ndarray],A:np.ndarray,dt:float=.001,T:float=.05,nspace:int=20,
              c:float=1.1,mu:float=2.):
    grid=np.arange(1,nspace+1)/(nspace+1);steps=round(T/dt)
    if abs(steps*dt-T)>1e-10:raise ValueError('T must divide dt')
    d,low,up=extract_blocks(A)
    if name in ('poisson_2d','helmholtz_2d'):
        raise ValueError('elliptic integration uses b=A@u_exact')
    if name in ('heat_1d','burgers_1d'):
        r=(.08 if name=='heat_1d' else .04)*dt*(nspace+1)**2
        x=manufacture(name,grid,t=0.)
        for step in range(1,steps+1):
            t=step*dt;rhs=x.copy()
            if name=='burgers_1d':
                old=np.r_[manufacture(name,np.array(0.),t=t-dt),x,manufacture(name,np.array(1.),t=t-dt)]
                rhs-=dt*x*(old[2:]-old[:-2])/(2/(nspace+1))
                rhs+=dt*forcing(name,grid,t=t-dt)
            else:rhs+=dt*forcing(name,grid,t=t)
            rhs[0]+=r*manufacture(name,np.array(0.),t=t)
            rhs[-1]+=r*manufacture(name,np.array(1.),t=t)
            x=cached_solve(low,up,inverse,rhs)
        return x
    if name=='klein_gordon_1d':
        u=manufacture('kg_u',grid,t=0.);v=manufacture('kg_v',grid,t=0.)
        for step in range(1,steps+1):
            t=step*dt;rhs=np.stack((u,v+dt*forcing('kg_v',grid,t=t,c=c,mu=mu)),axis=1).ravel()
            rhs[-1]+=dt*c*c*(nspace+1)**2*manufacture('kg_u',np.array(1.),t=t)
            state=cached_solve(low,up,inverse,rhs).reshape(nspace,2)
            u,v=state.T
        return np.stack((u,v),axis=1).ravel()
    raise KeyError(name)
