"""Prepare tables and figure data from the experiment records."""

from __future__ import annotations

from pathlib import Path

from reconstruct_dirac3_pde import regenerate

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def _write(df: pd.DataFrame, name: str) -> None:
    """Write one processed CSV with a stable column order supplied by the caller."""
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def build_k_sweep() -> None:
    """Create the seven-column convergence table used by Figure 2."""
    df = pd.read_csv(RAW / "k_sweep" / "k_sweep_results.csv")

    poisson = (
        df[df["pde"] == "poisson_2d"]
        .set_index("K")
        .sort_index()
    )
    kg = (
        df[df["pde"] == "klein_gordon_1d"]
        .set_index("K")
        .sort_index()
    )

    k_values = sorted(set(poisson.index) & set(kg.index))
    compact = pd.DataFrame(
        {
            "K": k_values,
            "poisson_solution": [poisson.loc[k, "rel_l2_vs_discrete_exact"] for k in k_values],
            "poisson_inverse": [poisson.loc[k, "mean_inv_fro_error"] for k in k_values],
            "poisson_energy": [poisson.loc[k, "mean_qubo_energy"] for k in k_values],
            "kg_solution": [kg.loc[k, "rel_l2_vs_dense_final"] for k in k_values],
            "kg_inverse": [kg.loc[k, "mean_inv_fro_error"] for k in k_values],
            "kg_energy": [kg.loc[k, "mean_qubo_energy"] for k in k_values],
        }
    )
    _write(compact, "k_sweep_compact.csv")


def build_large_block() -> None:
    """Copy the authoritative large-block summary into the processed-data layer."""
    df = pd.read_csv(RAW / "large_block" / "largeB_poisson_lineblock_summary.csv")
    _write(df, "largeB.csv")


def build_b8_optimizer() -> None:
    """Create the method and full-solve tables used in Section 5."""
    methods = pd.read_csv(RAW / "b8_optimizer" / "optimizer_summary_by_method.csv")
    full = pd.read_csv(RAW / "b8_optimizer" / "full_cached_solve_summary.csv")
    _write(methods, "b8_optimizer_methods.csv")
    _write(full, "b8_full_solve.csv")

    labels = {
        "rounded_reference": "Rounded",
        "greedy_from_rounded": "Greedy",
        "cold_SA_plus_greedy": "Cold SA",
        "warm_SA_plus_greedy": "Warm SA",
    }
    order = list(labels)
    plot = methods.set_index("method").loc[order].reset_index()
    plot = pd.DataFrame(
        {
            "short": plot["method"].map(labels),
            "median_objective_ratio_to_rounded": plot["median_objective_ratio_to_rounded"],
            "mean_objective_ratio_to_rounded": plot["mean_objective_ratio_to_rounded"],
            "median_residual_norm": plot["median_residual_norm"],
            "max_residual_norm": plot["max_residual_norm"],
            "median_column_error": plot["median_column_error"],
        }
    )
    _write(plot, "b8_optimizer_plot.csv")


def _poisson_slices(flat: pd.DataFrame) -> pd.DataFrame:
    """Convert the line-ordered 20-entry Poisson vector into two 10-point slices."""
    rows: list[dict[str, float | int]] = []
    for slice_id, offset in ((1, 0), (2, 1)):
        selected = flat.iloc[offset::2].reset_index(drop=True)
        for x_index, row in enumerate(selected.itertuples(index=False), start=1):
            rows.append(
                {
                    "x_index": x_index,
                    "slice": slice_id,
                    "dense": row.dense_ground_truth,
                    "hardware": row.qci_quantum,
                    "abs_error": row.abs_error,
                }
            )
    return pd.DataFrame(rows)


def build_dirac3() -> None:
    """Create the hardware-audit and end-to-end reconstruction CSVs."""
    raw = RAW / "dirac3"

    independent = regenerate(raw=raw, out=OUT, write=True, strict=True)
    heat = independent["fields"]["dirac_heat.csv"]
    poisson = independent["fields"]["dirac_poisson.csv"]
    kg = independent["fields"]["dirac_kg.csv"]
    _write(heat, "dirac_heat.csv")
    _write(poisson, "dirac_poisson.csv")
    _write(kg, "dirac_kg.csv")
    _write(_poisson_slices(poisson), "dirac_poisson_slices.csv")

    audit = pd.read_csv(raw / "mapped_60_instances.csv")
    pde_codes = {"Heat 1D": 1, "Poisson 2D": 2, "Klein-Gordon 1D": 3}

    scatter = pd.DataFrame(
        {
            "idx": np.arange(1, len(audit) + 1),
            "pde_name": audit["pde_name"],
            "pde_code": audit["pde_name"].map(pde_codes),
            "normalized_objective_gap": audit["normalized_objective_gap"],
            "local_inverse_residual_dirac": audit["local_inverse_residual_dirac"],
            "residual_ratio": audit["residual_ratio"],
            "exact_ground_state_match": audit["exact_ground_state_match"],
        }
    )
    _write(scatter, "dirac_scatter.csv")

    reconstructed = pd.read_csv(raw / "reconstructed_60_instances.csv")
    decoded = reconstructed[
        [
            "qci_id",
            "hw_best_decoded_y0",
            "hw_best_decoded_y1",
            "hw_exact_decoded_y0",
            "hw_exact_decoded_y1",
        ]
    ]
    mapped = audit.merge(decoded, on="qci_id", how="left")
    mapped = pd.DataFrame(
        {
            "qci_id": mapped["qci_id"],
            "pde_name": mapped["pde_name"],
            "block": mapped["block"],
            "column": mapped["column"],
            "unique_id": mapped["unique_id"],
            "exact_hit": mapped["exact_ground_state_match"],
            "exact_resid": mapped["local_inverse_residual_exact"],
            "hw_resid": mapped["local_inverse_residual_dirac"],
            "ratio": mapped["residual_ratio"],
            "hw_y0": mapped["hw_best_decoded_y0"],
            "hw_y1": mapped["hw_best_decoded_y1"],
            "exact_y0": mapped["hw_exact_decoded_y0"],
            "exact_y1": mapped["hw_exact_decoded_y1"],
            "dy_norm": mapped["dy_norm"],
            "energy_gap": mapped["absolute_objective_gap"],
        }
    )
    _write(mapped, "dirac_mapped.csv")

    pde_solution = independent["summary"].set_index("pde")
    rows: list[dict[str, object]] = []
    for pde_name, group in audit.groupby("pde_name", sort=False):
        solution_row = pde_solution.loc[pde_name]
        rows.append(
            {
                "pde_name": pde_name,
                "instances": len(group),
                "logical_qubo_variables": int(group["logical_qubo_variables"].iloc[0]),
                "qubo_terms": int(group["qubo_terms"].iloc[0]),
                "qubo_density": float(group["qubo_density"].iloc[0]),
                "exact_matches": int(group["exact_ground_state_match"].sum()),
                "exact_match_fraction": float(group["exact_ground_state_match"].mean()),
                "mean_abs_objective_gap": float(group["absolute_objective_gap"].mean()),
                "median_abs_objective_gap": float(group["absolute_objective_gap"].median()),
                "max_abs_objective_gap": float(group["absolute_objective_gap"].max()),
                "mean_norm_objective_gap": float(group["normalized_objective_gap"].mean()),
                "median_norm_objective_gap": float(group["normalized_objective_gap"].median()),
                "max_norm_objective_gap": float(group["normalized_objective_gap"].max()),
                "mean_residual_ratio": float(group["residual_ratio"].mean()),
                "median_residual_ratio": float(group["residual_ratio"].median()),
                "max_residual_ratio": float(group["residual_ratio"].max()),
                "median_local_residual": float(group["local_inverse_residual_dirac"].median()),
                "max_local_residual": float(group["local_inverse_residual_dirac"].max()),
                "pde_relative_error": float(solution_row["relative_error"]),
                "pde_relative_residual": float(solution_row["relative_residual"]),
            }
        )

    summary = pd.DataFrame(rows)
    _write(summary, "dirac_summary.csv")

    short = {"Heat 1D": "Heat", "Poisson 2D": "Poisson", "Klein-Gordon 1D": "Klein--Gordon"}
    plot_summary = pd.DataFrame(
        {
            "short": summary["pde_name"].map(short),
            "exact_match_fraction": summary["exact_match_fraction"],
            "mean_residual_ratio": summary["mean_residual_ratio"],
            "max_residual_ratio": summary["max_residual_ratio"],
            "median_local_residual": summary["median_local_residual"],
            "max_local_residual": summary["max_local_residual"],
            "pde_relative_error": summary["pde_relative_error"],
            "pde_relative_residual": summary["pde_relative_residual"],
        }
    )
    _write(plot_summary, "dirac_plot_summary.csv")


def build_multiscale_figure3() -> None:
    """Derive Table 3 and Figure 3 data from the five-PDE experiments."""
    reconstruction = ROOT / "finite_bit"
    source = reconstruction / "outputs"
    new = pd.read_csv(source / "figure3_multiscale.csv")
    fixed = pd.read_csv(source / "figure3_control.csv")
    order = ["heat_1d", "burgers_1d", "poisson_2d", "helmholtz_2d", "klein_gordon_1d"]
    labels = ["Heat", "Burgers", "Poisson", "Helmholtz", "Klein--Gordon"]
    assert new["pde"].tolist() == fixed["pde"].tolist() == order
    assert new["calls"].sum() == 480
    assert new["nonzero_correction_updates"].sum() == 360
    assert new["clipped_continuous_corrections"].sum() == 0
    assert (new["scale_policy"] == "gamma_p=gamma0/8**p").all()
    assert (new["logical_bits_per_call"] == 24).all()

    plot = pd.DataFrame({
        "short": labels,
        "rel_dense": new["rel_dense"],
        "max_inv_res": new["max_inv_res"],
        "baseline_rel_dense": fixed["rel_dense"],
        "baseline_max_inv_res": fixed["max_inv_res"],
    })
    _write(plot, "figure3_multiscale_plot.csv")
    _write(new, "figure3_multiscale_table.csv")

    def sci_tex(value: float) -> str:
        if value == 0:
            return "$0$"
        power = int(np.floor(np.log10(abs(float(value)))))
        return f"\${float(value) / 10**power:.2f}\\times10^{{{power}}}\$".replace("\\$", "$")

    rows = []
    for r in new.to_dict("records"):
        row = [
            str(r["label"]), f'{float(r["gamma0"]):.2f}', str(int(r["N"])),
            f'{float(r["cond_A"]):.2f}', sci_tex(r["rel_dense"]),
            sci_tex(r["r_inf"]), sci_tex(r["max_inv_res"]),
            f'{float(r["max_schur_cond"]):.2f}',
            f'{int(r["calls"])} / {int(r["nonzero_correction_updates"])}',
        ]
        rows.append(" & ".join(row) + r" \\")
    (OUT / "table3_multiscale_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_k_sweep()
    build_multiscale_figure3()
    build_large_block()
    build_b8_optimizer()
    build_dirac3()
    print(f"Processed analysis data regenerated in: {OUT}")


if __name__ == "__main__":
    main()
