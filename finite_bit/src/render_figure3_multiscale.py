"""Render the five-PDE field and correction diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';FIG=ROOT/'figures';MAN=ROOT/'manuscript'
DISPLAY=('Heat','Burgers','Poisson','Helmholtz','Klein–Gordon')
ORDER=('heat_1d','burgers_1d','poisson_2d','helmholtz_2d','klein_gordon_1d')


def read(path):
    return list(csv.DictReader(Path(path).open(newline='',encoding='utf8')))


def sci(x, digits=2):
    x=float(x)
    if x==0:return '$0$'
    power=int(np.floor(np.log10(abs(x))))
    return '$'+f'{x/10**power:.{digits}f}\\times10^{{{power}}}'+'$'


def export(fig,basename):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(basename+'.png'),dpi=220,bbox_inches='tight')
    fig.savefig(FIG/(basename+'.svg'),bbox_inches='tight')
    plt.close(fig)


def main():
    results=read(OUT/'figure3_multiscale.csv');control=read(OUT/'figure3_control.csv')
    events=read(OUT/'figure3_multiscale_stage_log.csv');fields=read(OUT/'figure3_multiscale_fields.csv')
    assert [r['pde'] for r in results]==list(ORDER) and len(events)==480 and len(fields)==120
    assert [r['pde'] for r in control]==list(ORDER)
    control_by={r['pde']:r for r in control}
    fig,(ax,ay)=plt.subplots(1,2,figsize=(12.8,4.7),layout='constrained')
    xs=np.arange(5)
    for panel, key, lab in ((ax,'rel_dense','Relative error vs dense numerical solve'),
                             (ay,'max_inv_res',r'Maximum $\|S_i\widehat{S_i^{-1}}-I\|_F$')):
        v=np.array([float(r[key]) for r in results]);controlv=np.array([float(control_by[r['pde']][key]) for r in results])
        panel.semilogy(xs,controlv,linestyle='None',marker='o',fillstyle='none',markersize=9,
                       markeredgewidth=1.5,label='Single-grid baseline (three zero corrections)')
        panel.semilogy(xs,v,linestyle='None',marker='D',markersize=7,
                       label='Predetermined 1/8 multiscale refinement')
        panel.set_xticks(xs,DISPLAY,rotation=22,ha='right');panel.set_xlim(-.45,4.45)
        panel.set_ylabel(lab);panel.grid(axis='y',which='both',alpha=.3)
        panel.set_ylim(min(np.r_[v,controlv])*.42,max(np.r_[v,controlv])*2.5)
        panel.legend(loc='best',fontsize=8)
        for i,y in enumerate(v):panel.annotate(f'{y:.1e}',(i,y),xytext=(0,-17),ha='center',fontsize=8,textcoords='offset points')
    ax.set_title('(a) End-to-end accuracy');ay.set_title('(b) Cached inverse quality')
    fig.suptitle(r'Five PDEs  |  $B=2$, $M=1$, $K=10$  |  $\gamma_p=\gamma_0 8^{-p}$, $p=0,1,2,3$')
    export(fig,'figure3_multiscale')

    stage_summ=[];fig,ax=plt.subplots(figsize=(10.2,5.0),layout='constrained')
    for row,label in zip(results,DISPLAY):
        group=[e for e in events if e['pde']==row['pde']]
        curves=[]
        for p in range(4):
            blockres=[]
            for block in range(1,int(row['N'])//2+1):
                a=np.array([float(e['post_column_residual']) for e in group if int(e['stage'])==p and int(e['block'])==block]);assert len(a)==2
                blockres.append(np.linalg.norm(a))
            v=float(max(blockres));curves.append(v)
            stage_summ.append(dict(pde=row['pde'],stage=p,gamma=float(row['gamma0'])/8**p,
                                  max_block_inverse_residual=v,
                                  nonzero_updates=sum((float(e['delta0'])!=0 or float(e['delta1'])!=0) for e in group if int(e['stage'])==p),
                                  clipped_audit_corrections=sum(e['continuous_correction_within_grid'].lower()!='true' for e in group if int(e['stage'])==p)))
        ax.semilogy(range(4),curves,marker='o',label=label)
        assert abs(curves[-1]-float(row['max_inv_res']))<2e-12
    ax.set_xticks(range(4),['Initial','Correction 1','Correction 2','Correction 3'])
    ax.set_ylabel('Maximum block inverse Frobenius residual')
    ax.set_title('Exact QUBO multiscale correction progressively refines cached Schur factors')
    ax.grid(axis='y',which='both',alpha=.3);ax.legend(ncol=2,fontsize=9,frameon=False)
    export(fig,'figure3_multiscale_stages')
    with (OUT/'figure3_multiscale_stage_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,stage_summ[0].keys());w.writeheader();w.writerows(stage_summ)

    fdict={n:[] for n in ORDER}
    for z in fields:fdict[z['pde']].append(z)
    fig,axs=plt.subplots(3,2,figsize=(12.2,12.0),layout='constrained')
    details={}
    for name,axis in zip(ORDER[:-1],axs.flat[:4]):
        row=sorted(fdict[name],key=lambda r:int(r['component_index']))
        f=np.array([float(r['multiscale']) for r in row]);d=np.array([float(r['dense_numerical']) for r in row]);a=np.array([float(r['manufactured']) for r in row])
        if name in ('heat_1d','burgers_1d'):
            xx=np.arange(1,21)/21
            axis.plot(xx,a,':',label='Manufactured analytical field',lw=1.8)
            axis.plot(xx,d,'-',label='Dense numerical',lw=1.8)
            axis.plot(xx,f,'--o',ms=2.5,label='Multiscale QUBO')
            axis.set_xlabel('Spatial coordinate x')
        else:
            xx=np.arange(1,11)/11
            for yi,style in ((0,'-'),(1,':')):
                axis.plot(xx,d[yi::2],ls=style,lw=1.8,label=f'Dense = manufactured y={(yi+1)/3:.2f}')
                axis.plot(xx,f[yi::2],'--o',ms=2.5,label=f'Multiscale y={(yi+1)/3:.2f}')
            assert max(abs(d-a))<2e-12
            axis.set_xlabel('Spatial coordinate x (10×2 grid)')
        err=float(np.linalg.norm(f-d)/np.linalg.norm(d));assert abs(err-float(next(r for r in results if r['pde']==name)['rel_dense']))<2e-12
        details[name]={'rel_dense':err,'max_abs_dense':float(np.max(abs(f-d)))}
        axis.set_title(f'{DISPLAY[ORDER.index(name)]}: relative error {err:.2e}')
        axis.grid(alpha=.2);axis.legend(fontsize=7)
    name='klein_gordon_1d';kg=sorted(fdict[name],key=lambda r:int(r['component_index']))
    f=np.array([float(r['multiscale']) for r in kg]);d=np.array([float(r['dense_numerical']) for r in kg]);a=np.array([float(r['manufactured']) for r in kg]);xx=np.arange(1,21)/21
    for v,label,axis in ((0,'u',axs[2,0]),(1,'v',axs[2,1])):
        axis.plot(xx,a[v::2],':',label='Manufactured analytical',lw=1.8)
        axis.plot(xx,d[v::2],'-',label='Dense numerical',lw=1.8)
        axis.plot(xx,f[v::2],'--o',label='Multiscale QUBO',ms=2.5)
        err=float(np.linalg.norm(f[v::2]-d[v::2])/np.linalg.norm(d[v::2]))
        assert abs(err-float(next(r for r in results if r['pde']==name)[label+'_rel_dense']))<2e-12
        details[name+'_'+label]={'rel_dense':err,'max_abs_dense':float(max(abs(f[v::2]-d[v::2])))}
        axis.set_title(f'Klein–Gordon {label}: relative error {err:.2e}')
        axis.set_xlabel('Spatial coordinate x');axis.grid(alpha=.2);axis.legend(fontsize=8)
    export(fig,'figure3_multiscale_ground_truth')

    plot=[dict(short=DISPLAY[i].replace('–','--'),rel_dense=r['rel_dense'],max_inv_res=r['max_inv_res'],
               baseline_rel_dense=control_by[r['pde']]['rel_dense'],baseline_max_inv_res=control_by[r['pde']]['max_inv_res']) for i,r in enumerate(results)]
    MAN.mkdir(parents=True,exist_ok=True)
    baseline_kg=read(OUT/'figure3_control_fields.csv')
    baseline_kg=sorted((r for r in baseline_kg if r['pde']=='klein_gordon_1d'),key=lambda r:int(r['component_index']))
    assert len(baseline_kg)==40
    bf=np.array([float(r['single_grid']) for r in baseline_kg]);bd=np.array([float(r['dense_numerical']) for r in baseline_kg])
    components=[dict(component=v,rel_dense=results[-1][v+'_rel_dense'],
                     rel_manufactured=results[-1][v+'_rel_exact'],
                     baseline_rel_dense=float(np.linalg.norm((bf-bd)[j::2])/np.linalg.norm(bd[j::2])))
                for v,j in (('u',0),('v',1))]
    for name,values in [('figure3_multiscale_plot.csv',plot),('figure3_multiscale_kg_components.csv',components)]:
        with (MAN/name).open('w',newline='') as ff:
            w=csv.DictWriter(ff,values[0].keys());w.writeheader();w.writerows(values)
    md=['| PDE | $\\gamma_0$ | $N$ | $\\kappa_2(A)$ | $e_{\\rm dense}$ | $r_\\infty$ | $\\rho_S$ | $\\max\\kappa_2(S_i)$ | Calls / nonzero corrections |',
        '|:--|--:|--:|--:|--:|--:|--:|--:|--:|']
    tex=[]
    for r in results:
        vals=(r['label'],f'{float(r["gamma0"]):.2f}',r['N'],f'{float(r["cond_A"]):.2f}',sci(r['rel_dense']),sci(r['r_inf']),sci(r['max_inv_res']),f'{float(r["max_schur_cond"]):.2f}',f'{r["calls"]} / {r["nonzero_correction_updates"]}')
        md.append('| '+' | '.join(vals).replace('--','–')+' |')
        tex.append(' & '.join(vals)+r' \\')
    (MAN/'TABLE3_NEW_MULTISCALE.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    (MAN/'TABLE3_NEW_MULTISCALE_rows.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    compare=['| PDE | Fixed-grid $e_{\rm dense}$ | Multiscale $e_{\rm dense}$ | Fixed-grid $\rho_S$ | Multiscale $\rho_S$ |',
             '|:--|--:|--:|--:|--:|']
    for r in results:
        o=control_by[r['pde']]
        compare.append('| '+' | '.join((r['label'].replace('--','–'),sci(o['rel_dense']),sci(r['rel_dense']),sci(o['max_inv_res']),sci(r['max_inv_res'])))+' |')
    (MAN/'TABLE3_BASELINE_VS_MULTISCALE.md').write_text('\n'.join(compare)+'\n',encoding='utf8')
    kgcomp=['| Klein–Gordon component | Fixed-grid relative error | Multiscale relative error |',
            '|:--|--:|--:|']
    for v in components:kgcomp.append('| '+' | '.join((v['component'],sci(v['baseline_rel_dense']),sci(v['rel_dense'])))+' |')
    (MAN/'TABLE3_KG_COMPONENTS.md').write_text('\n'.join(kgcomp)+'\n',encoding='utf8')
    (FIG/'figure3_multiscale_ground_truth_metrics.json').write_text(json.dumps({'method':'predetermined multiscale correction schedule','fields':details},indent=2)+'\n')
    print('Rendered: main Figure 3, stage diagnostics, 6-panel field overlay, Table 3 and manuscript plot data.')
    return results

if __name__=='__main__':main()
