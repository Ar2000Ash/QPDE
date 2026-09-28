"""Compare independent large-block and optimizer runs with published tables.

Elapsed times are hardware-dependent and are excluded from numerical comparisons.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def check_table(actual: Path, reference: Path, *, key: str) -> None:
    observed = pd.read_csv(actual).set_index(key).sort_index()
    expected = pd.read_csv(reference).set_index(key).sort_index()

    assert observed.index.equals(expected.index), (actual, "row identity")
    exclude = {"schur_factor_generation_time_s", "runtime_seconds_for_greedy_polish"}
    columns = [column for column in expected if column not in exclude]
    assert set(columns).issubset(observed.columns), (actual, "missing columns")

    for column in columns:
        current = observed[column]
        recorded = expected[column]
        if pd.api.types.is_numeric_dtype(recorded):
            np.testing.assert_allclose(
                current.to_numpy(dtype=float),
                recorded.to_numpy(dtype=float),
                rtol=2e-6,
                atol=1e-11,
                equal_nan=True,
                err_msg=f"{actual.name}: {column}",
            )
        else:
            assert current.fillna("").astype(str).equals(
                recorded.fillna("").astype(str)
            ), (actual.name, column)
    print(f"PASS: {actual.name} matches {len(observed)} source rows")


def run(large_block: Path, optimizer: Path) -> None:
    check_table(
        large_block,
        RAW / "large_block" / "largeB_poisson_lineblock_summary.csv",
        key="B",
    )
    check_table(
        optimizer / "summary_by_method.csv",
        RAW / "b8_optimizer" / "optimizer_summary_by_method.csv",
        key="method",
    )
    check_table(
        optimizer / "full_cached_solve_summary.csv",
        RAW / "b8_optimizer" / "full_cached_solve_summary.csv",
        key="method",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--large",
        type=Path,
        default=ROOT / "outputs" / "large_block" / "summary.csv",
    )
    parser.add_argument(
        "--b8",
        type=Path,
        default=ROOT / "outputs" / "b8_optimizer",
    )
    args = parser.parse_args()
    run(args.large, args.b8)
