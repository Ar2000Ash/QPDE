"""Recompute the precision sweep and fitted convergence slopes."""
from pathlib import Path
import sys,csv,math,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from figure2_exact_reconstruction import all_rows
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/'data'/'raw'/'k_sweep'/'k_sweep_results.csv'
CHECK_KEYS=('B','M','gamma','q_per_scalar','qubo_bits','candidates_per_column','num_qubo_column_solves',
    'rel_l2_vs_discrete_exact','rel_l2_vs_dense','linf_vs_discrete_exact','residual_norm','relative_residual',
    'exact_block_vs_dense_rel_l2','mean_qubo_energy','max_qubo_energy','mean_column_residual',
    'max_column_residual','mean_inv_fro_error','max_inv_fro_error','mean_inv_residual_fro',
    'max_inv_residual_fro','max_schur_condition','mean_theory_bound_fro','max_theory_bound_fro',
    'fixed_point_range_low','fixed_point_range_high','exact_schur_inverse_min_entry',
    'exact_schur_inverse_max_entry','clipping_risk','num_time_steps','rel_l2_vs_dense_final',
    'rel_l2_vs_analytic_final','dense_rel_l2_vs_analytic_final','linf_vs_dense_final')

CORE_POISSON=[.07769322801868642,.009424208808900948,.017766745465914602,.004878427095471854,.003678904302493722,.0015450496701561727,.0006943865829574462]
CORE_KG=[.3151631078613268,.15691506279099907,.13275670849260268,.013998470232349357,.02367022803569749,.0044887266357106345,.006669423288797638]

def check():
    generated=all_rows();assert len(generated)==14
    if not ARCHIVE.exists():
        for r in generated:
            expected=(CORE_POISSON if r['pde']=='poisson_2d' else CORE_KG)[r['K']-4]
            actual=r['rel_l2_vs_discrete_exact'] if r['pde']=='poisson_2d' else r['rel_l2_vs_dense_final']
            assert abs(actual-expected)<2e-12,(r['pde'],r['K'],actual,expected)
        print('PASS: 14/14 Figure 2 core solution errors reproduced; complete source CSV unavailable for full-column comparison')
        return
    raw=list(csv.DictReader(ARCHIVE.open(newline='')))
    table={(r['pde'],int(r['K'])):r for r in raw}
    assert len(raw)==14 and len(table)==14
    comparisons=0;worst=('none',0.);errors=[]
    for x in generated:
        key=x['pde'],x['K'];old=table[key]
        for field in CHECK_KEYS:
            if field not in x or not old.get(field):continue
            s=old[field]
            if field=='clipping_risk':
                ok=(x[field]==(s.lower()=='true'));diff=0. if ok else 1.
            else:
                val=float(s);cur=float(x[field]);diff=abs(val-cur)
                ok=diff<=2e-11+1e-10*abs(val)
            comparisons+=1
            if diff>worst[1]:worst=(str(key)+':'+field,diff)
            if not ok:errors.append({'key':key,'field':field,'recorded':s,'recomputed':x[field],'abs_diff':diff})
    slope_file=ARCHIVE.parent/'convergence_slopes.csv'
    slope_diffs=[]
    if slope_file.exists():
        for row in csv.DictReader(slope_file.open(newline='')):
            pde=row['pde'];v=sorted((x for x in generated if x['pde']==pde),key=lambda x:x['K'])
            for field,original in [('mean_inv_fro_error','slope_log2_mean_inverse_error_vs_K'),
                                   ('mean_qubo_energy','slope_log2_mean_qubo_energy_vs_K')]:
                slope=float(np.polyfit([q['K'] for q in v],np.log2([q[field] for q in v]),1)[0])
                slope_diffs.append(abs(slope-float(row[original])))
            field='rel_l2_vs_discrete_exact' if pde=='poisson_2d' else 'rel_l2_vs_dense_final'
            slope=float(np.polyfit([q['K'] for q in v],np.log2([q[field] for q in v]),1)[0])
            slope_diffs.append(abs(slope-float(row['slope_log2_error_vs_K'])))
        assert max(slope_diffs)<1e-10,slope_diffs
    print(json.dumps({'complete_rows':len(generated),'numeric_columns_compared':comparisons,'slope_columns_checked':len(slope_diffs),
                      'worst_absolute_difference':worst,'disagreements':errors[:15],
                      'accepted':not errors},indent=2))
    assert not errors,f'Figure 2 mismatch in {len(errors)} columns'
if __name__=='__main__':check()
