"""Recompute Table 3, field values, and per-stage inverse corrections."""
import csv
import json
import tempfile
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from figure3_multiscale import generate
from render_figure3_multiscale import main as render


def run():
    original=ROOT/'outputs'
    with tempfile.TemporaryDirectory() as td:
        newpath=Path(td)
        generate(newpath)
        for fn in ('figure3_multiscale.csv','figure3_multiscale_stage_log.csv','figure3_multiscale_fields.csv'):
            actual=list(csv.DictReader((newpath/fn).open(newline='')))
            stored=list(csv.DictReader((original/fn).open(newline='')))
            assert len(actual)==len(stored),(fn,len(actual),len(stored))
            for i,(row,ref) in enumerate(zip(actual,stored)):
                assert row.keys()==ref.keys()
                for k in row:
                    if k=='precompute_cpu_reduced_seconds':continue
                    if row[k] in ('True','False') or not row[k] or not (row[k][0].isdigit() or row[k][0] in '+-.'):
                        assert row[k]==ref[k],(fn,i,k,row[k],ref[k]);continue
                    try:a=float(row[k]);b=float(ref[k])
                    except ValueError:assert row[k]==ref[k];continue
                    assert np.isclose(a,b,rtol=2e-9,atol=5e-13),(fn,i,k,a,b)
        d=json.loads((original/'figure3_multiscale_metadata.json').read_text())
        assert d['total_calls']==480 and d['total_correction_steps']==360 and d['clipped_audit_corrections']==0
    render() # Also verifies table, per-component field and residual-stage consistency.
    print('PASS: committed multiscale tables, stage bitstrings, field curves, and figure/LaTeX sources agree with fresh five-PDE rerun')

if __name__=='__main__':run()
