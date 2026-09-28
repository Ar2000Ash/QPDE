#!/usr/bin/env python3
"""Print a compact audit of the archived QCI Dirac-3 hardware campaign.

This script uses only stored CSV/JSON evidence.  It never contacts QCI services and
requires no credentials.  Its purpose is to reproduce the hardware accounting and the
per-PDE exact-hit/residual summaries reported in the study.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from check_dirac3_decode import check_archive

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dirac3"


def main() -> None:
    check_archive()
    mapped = pd.read_csv(RAW / "mapped_60_instances.csv")
    unique = pd.read_csv(RAW / "unique_jobs.csv")
    samples = pd.read_csv(RAW / "returned_samples.csv")

    print("Dirac-3 campaign audit")
    print("----------------------")
    print(f"Mapped inverse-column instances : {len(mapped)}")
    print(f"Unique submitted QUBOs          : {len(unique)}")
    print(f"Mapped exact finite-bit hits    : {int(mapped['exact_ground_state_match'].sum())}/{len(mapped)}")
    print(f"Unique-QUBO exact hits          : {int(unique['strict_exact_hit_any'].sum())}/{len(unique)}")
    print(f"Metered device usage           : {int(unique['device_usage_s'].sum())} s")
    print(f"Stored returned unique samples : {len(samples)} rows")

    summary = (
        mapped.groupby("pde_name", sort=False)
        .agg(
            instances=("qci_id", "count"),
            exact_hits=("exact_ground_state_match", "sum"),
            mean_residual_ratio=("residual_ratio", "mean"),
            max_local_residual=("local_inverse_residual_dirac", "max"),
            max_normalized_gap=("normalized_objective_gap", "max"),
        )
    )
    print("\nPer-PDE mapped-instance summary:\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
