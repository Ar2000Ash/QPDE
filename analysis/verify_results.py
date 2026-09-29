"""Check numerical tables and hardware statistics."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from check_dirac3_decode import check_archive

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def close(actual: float, expected: float, *, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    """Raise a readable error when a scalar differs from its archived reference."""
    if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def check_k_sweep() -> None:
    slopes = pd.read_csv(RAW / "k_sweep" / "convergence_slopes.csv").set_index("pde")
    close(slopes.loc["poisson_2d", "slope_log2_error_vs_K"], -0.9966780666375166)
    close(slopes.loc["poisson_2d", "slope_log2_mean_inverse_error_vs_K"], -1.0354225709372773)
    close(slopes.loc["poisson_2d", "slope_log2_mean_qubo_energy_vs_K"], -1.9321661980343798)
    close(slopes.loc["klein_gordon_1d", "slope_log2_error_vs_K"], -1.0510673348750195)
    close(slopes.loc["klein_gordon_1d", "slope_log2_mean_inverse_error_vs_K"], -1.079967985639443)
    close(slopes.loc["klein_gordon_1d", "slope_log2_mean_qubo_energy_vs_K"], -1.9397020035662988)


def check_five_pde() -> None:
    """Verify the five-PDE multiscale Table 3 results."""
    d = pd.read_csv(PROCESSED / "figure3_multiscale_table.csv").set_index("pde")
    expected = {
        "heat_1d": (1.3367080088581643e-05, 8.341045249140769e-07, 80, 60),
        "burgers_1d": (1.0986839891150165e-05, 7.141050527137744e-07, 80, 60),
        "poisson_2d": (3.0522183023661557e-06, 3.4461308764777923e-06, 80, 60),
        "helmholtz_2d": (6.884600474089066e-06, 3.826884827787936e-06, 80, 60),
        "klein_gordon_1d": (2.3027081329091724e-05, 8.610285564378199e-07, 160, 120),
    }
    assert set(d.index) == set(expected)
    for pde, (error, residual, calls, updates) in expected.items():
        close(d.loc[pde, "rel_dense"], error)
        close(d.loc[pde, "max_inv_res"], residual)
        assert int(d.loc[pde, "calls"]) == calls
        assert int(d.loc[pde, "nonzero_correction_updates"]) == updates
        assert int(d.loc[pde, "clipped_continuous_corrections"]) == 0
    assert int(d["calls"].sum()) == 480


def check_large_block() -> None:
    df = pd.read_csv(PROCESSED / "largeB.csv").set_index("B")
    assert list(df.index.astype(int)) == [2, 4, 8, 16, 32]
    assert list(df["encoded_bits_per_inverse_column"].astype(int)) == [24, 48, 96, 192, 384]
    assert np.allclose(df["saturation_fraction"], 0.0)
    close(df.loc[8, "relative_error_vs_dense"], 0.0014550918390404603)
    close(df.loc[32, "relative_residual"], 0.11971484580699328)


def check_b8_optimizer() -> None:
    methods = pd.read_csv(PROCESSED / "b8_optimizer_methods.csv").set_index("method")
    full = pd.read_csv(PROCESSED / "b8_full_solve.csv").set_index("method")
    assert int(methods.loc["rounded_reference", "instances"]) == 32
    close(methods.loc["cold_SA_plus_greedy", "median_objective_ratio_to_rounded"], 37640.026040103505)
    close(methods.loc["warm_SA_plus_greedy", "mean_objective_ratio_to_rounded"], 0.8812463458662044)
    close(full.loc["rounded_cached_inverse", "relative_error_vs_dense"], 0.0014550918390404603)
    close(full.loc["greedy_polished_cached_inverse", "relative_error_vs_dense"], 0.0006401607568273323)
    close(full.loc["greedy_polished_cached_inverse", "columns_improved_fraction"], 0.48333333333333334)
    close(full.loc["greedy_polished_cached_inverse", "relative_residual"], 0.013703790913998436)


def check_dirac3() -> None:
    raw = RAW / "dirac3"
    mapped = pd.read_csv(raw / "mapped_60_instances.csv")
    unique = pd.read_csv(raw / "unique_jobs.csv")
    samples = pd.read_csv(raw / "returned_samples.csv")
    summary = pd.read_csv(PROCESSED / "dirac_summary.csv").set_index("pde_name")

    assert len(mapped) == 60
    assert int(mapped["exact_ground_state_match"].sum()) == 31
    assert len(unique) == 34
    assert int(unique["strict_exact_hit_any"].sum()) == 17
    assert int(unique["device_usage_s"].sum()) == 558
    assert len(samples) == 457

    close(summary.loc["Heat 1D", "exact_match_fraction"], 0.20)
    close(summary.loc["Poisson 2D", "exact_match_fraction"], 0.40)
    close(summary.loc["Klein-Gordon 1D", "exact_match_fraction"], 0.95)
    close(summary.loc["Heat 1D", "pde_relative_error"], 3.2391279008389326e-04)
    close(summary.loc["Poisson 2D", "pde_relative_error"], 1.970816149365653e-03)
    close(summary.loc["Klein-Gordon 1D", "pde_relative_error"], 2.9703911071363637e-04)


def main() -> None:
    check_k_sweep()
    check_five_pde()
    check_large_block()
    check_b8_optimizer()
    check_dirac3()
    check_archive()
    print("All numerical checks passed.")


if __name__ == "__main__":
    main()
