"""Check recorded QCI finite-bit reference minima against the exact grid."""
from pathlib import Path
import sys,csv,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exact_oracle import reduced_exact,decode_bits,qubo_upper
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT.parent/'data'/'raw'/'dirac3'/'reconstructed_60_instances.csv'

def check():
    rows=list(csv.DictReader(RAW.open()));seen=set();errors=[]; second_errors=[]; matched=0; mismatches=[]
    for row in rows:
        # Repeated QCI IDs share the same QUBO; only independent representative ones.
        uid=(row['S00'],row['S01'],row['S10'],row['S11'],row['gamma'],row['column'])
        if uid in seen: continue
        seen.add(uid)
        S=np.array([[float(row['S00']),float(row['S01'])],[float(row['S10']),float(row['S11'])]])
        col=int(row['column']);res=reduced_exact(S,np.eye(2)[:,col],float(row['gamma']),M=int(row['M']),K=int(row['K']))
        exp=str(row['global_min_bitstring']).zfill(24)
        energy=float(row['global_min_energy'])
        if res.bitstring==exp:matched+=1
        else:mismatches.append([row['qci_id'],exp,res.bitstring,float(energy),res.residual_squared])
        errors.append(abs(res.residual_squared-energy))
        second_errors.append(abs(res.second_residual_squared-float(row['second_energy'])))
    print(json.dumps({'distinct_S_gamma_column':len(seen),'exact_bitstrings_matched':matched,'bitstring_mismatches':mismatches,'max_abs_residual_squared_error':max(errors),'max_abs_second_best_energy_error':max(second_errors)},indent=2))
    assert len(seen)==34 and not mismatches and max(errors)<2e-12 and max(second_errors)<2e-12
    return len(seen),len(mismatches),max(errors)
if __name__=='__main__':check()
