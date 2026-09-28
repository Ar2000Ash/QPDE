#!/usr/bin/env python3
"""Re-run the 96-bit B=8 optimizer experiment from the reported study.

The experiment compares four states/strategies on 32 representative inverse-column
QUBOs: direct fixed-point rounding, greedy polishing, cold-start simulated annealing
plus greedy polishing, and warm-start simulated annealing plus greedy polishing.  It
then applies greedy polishing to all 960 inverse columns and evaluates the downstream
Poisson solve.

The random seeds and search budgets match the archived experiment.  No exact 96-bit
ground-state claim is made or implied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpde_schur.binary import (
    decode_bits,
    greedy_descent,
    make_bit_layout,
    quantize_vector_to_bits,
    residual_energy,
    simulated_annealing,
)
from qpde_schur.core import (
    block_schur_solve,
    build_dense_block_matrix,
    gamma_from_block,
    poisson_line_blocks,
    schur_inverse_factors,
    smooth_rhs,
)


N = 960
B = 8
M = 1
K = 10
N_BLOCKS = N // B
REPRESENTATIVE_BLOCKS_1BASED = (1, 30, 60, 120)
METHOD_ORDER = (
    "rounded_reference",
    "greedy_from_rounded",
    "cold_SA_plus_greedy",
    "warm_SA_plus_greedy",
)


def run_experiment(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the representative and full-factor optimizer studies and save CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()

    layout = make_bit_layout(B, M, K)
    D, L, U = poisson_line_blocks(B)
    schur_blocks, exact_inverse_blocks = schur_inverse_factors(D, L, U, N_BLOCKS)
    gammas = [
        gamma_from_block(S, S_inv, M)[0]
        for S, S_inv in zip(schur_blocks, exact_inverse_blocks)
    ]

    # -------------------------------------------------------------------------
    # Representative 32-QUBO comparison: four Schur blocks x eight columns.
    # -------------------------------------------------------------------------
    representative_rows: list[dict[str, object]] = []
    for block_1based in REPRESENTATIVE_BLOCKS_1BASED:
        block_index = block_1based - 1
        S = schur_blocks[block_index]
        S_inv = exact_inverse_blocks[block_index]
        gamma = gammas[block_index]

        for column in range(B):
            y_exact = S_inv[:, column]
            q_round, y_round, saturated = quantize_vector_to_bits(y_exact, gamma, layout)
            e_round = residual_energy(S, column, q_round, gamma, layout)

            q_greedy, e_greedy, greedy_flips, _ = greedy_descent(
                S, column, q_round, gamma, layout, max_sweeps=8
            )
            y_greedy = decode_bits(q_greedy, gamma, layout)

            q_cold, e_cold, cold_accepts, cold_greedy_flips = simulated_annealing(
                S,
                column,
                gamma,
                layout,
                q_start=None,
                sweeps=140,
                restarts=2,
                seed=100000 + 1000 * block_1based + column,
            )
            y_cold = decode_bits(q_cold, gamma, layout)

            q_warm, e_warm, warm_accepts, warm_greedy_flips = simulated_annealing(
                S,
                column,
                gamma,
                layout,
                q_start=q_round,
                sweeps=120,
                restarts=2,
                seed=200000 + 1000 * block_1based + column,
                warm_perturb=0.015,
            )
            y_warm = decode_bits(q_warm, gamma, layout)

            candidates = [
                ("rounded_reference", q_round, y_round, e_round, 0, 0),
                ("greedy_from_rounded", q_greedy, y_greedy, e_greedy, 0, greedy_flips),
                ("cold_SA_plus_greedy", q_cold, y_cold, e_cold, cold_accepts, cold_greedy_flips),
                ("warm_SA_plus_greedy", q_warm, y_warm, e_warm, warm_accepts, warm_greedy_flips),
            ]
            best_observed = min(item[3] for item in candidates)

            for method, q, y_hat, energy, accepts, final_flips in candidates:
                representative_rows.append(
                    {
                        "B": B,
                        "N": N,
                        "block_1based": block_1based,
                        "column": column,
                        "method": method,
                        "logical_bits": layout.n_bits,
                        "qubo_terms": layout.n_bits * (layout.n_bits + 1) // 2,
                        "gamma_i": gamma,
                        "objective_residual_squared": energy,
                        "local_residual_norm": math.sqrt(max(energy, 0.0)),
                        "objective_ratio_to_rounded": energy / max(e_round, 1e-300),
                        "objective_ratio_to_best_observed": energy / max(best_observed, 1e-300),
                        "decoded_column_error_norm": float(np.linalg.norm(y_hat - y_exact)),
                        "bit_hamming_vs_rounded": int(np.sum(q != q_round)),
                        "accepted_SA_flips": int(accepts),
                        "final_greedy_flips": int(final_flips),
                        "saturation_entries": saturated if method == "rounded_reference" else "",
                        "bitstring": "".join(str(int(bit)) for bit in q),
                    }
                )

    representative = pd.DataFrame(representative_rows)
    representative.to_csv(output_dir / "representative_instances.csv", index=False)

    summary = (
        representative.groupby("method")
        .agg(
            instances=("objective_residual_squared", "count"),
            median_objective_ratio_to_rounded=("objective_ratio_to_rounded", "median"),
            mean_objective_ratio_to_rounded=("objective_ratio_to_rounded", "mean"),
            max_objective_ratio_to_rounded=("objective_ratio_to_rounded", "max"),
            median_residual_norm=("local_residual_norm", "median"),
            mean_residual_norm=("local_residual_norm", "mean"),
            max_residual_norm=("local_residual_norm", "max"),
            median_column_error=("decoded_column_error_norm", "median"),
            mean_column_error=("decoded_column_error_norm", "mean"),
            max_column_error=("decoded_column_error_norm", "max"),
            median_hamming_vs_rounded=("bit_hamming_vs_rounded", "median"),
        )
        .reset_index()
    )
    order_map = {method: i for i, method in enumerate(METHOD_ORDER)}
    summary["_order"] = summary["method"].map(order_map)
    summary = summary.sort_values("_order").drop(columns="_order")
    summary.to_csv(output_dir / "summary_by_method.csv", index=False)

    # -------------------------------------------------------------------------
    # Full 960-column greedy polish and downstream PDE solve.
    # -------------------------------------------------------------------------
    rounded_inverse_blocks: list[np.ndarray] = []
    greedy_inverse_blocks: list[np.ndarray] = []
    full_rows: list[dict[str, object]] = []

    polish_start = time.perf_counter()
    for block_1based, (S, S_inv, gamma) in enumerate(
        zip(schur_blocks, exact_inverse_blocks, gammas), start=1
    ):
        rounded_block = np.zeros((B, B), dtype=float)
        greedy_block = np.zeros((B, B), dtype=float)

        for column in range(B):
            q_round, y_round, _ = quantize_vector_to_bits(S_inv[:, column], gamma, layout)
            e_round = residual_energy(S, column, q_round, gamma, layout)
            q_greedy, e_greedy, greedy_flips, _ = greedy_descent(
                S, column, q_round, gamma, layout, max_sweeps=8
            )
            y_greedy = decode_bits(q_greedy, gamma, layout)

            rounded_block[:, column] = y_round
            greedy_block[:, column] = y_greedy
            full_rows.append(
                {
                    "block_1based": block_1based,
                    "column": column,
                    "gamma_i": gamma,
                    "rounded_objective": e_round,
                    "greedy_objective": e_greedy,
                    "objective_improvement_fraction": (e_round - e_greedy) / max(e_round, 1e-300),
                    "rounded_residual_norm": math.sqrt(max(e_round, 0.0)),
                    "greedy_residual_norm": math.sqrt(max(e_greedy, 0.0)),
                    "greedy_flips": greedy_flips,
                    "hamming_vs_rounded": int(np.sum(q_greedy != q_round)),
                }
            )

        rounded_inverse_blocks.append(rounded_block)
        greedy_inverse_blocks.append(greedy_block)

    polish_seconds = time.perf_counter() - polish_start
    full_columns = pd.DataFrame(full_rows)
    full_columns.to_csv(output_dir / "full_greedy_polish_all_columns.csv", index=False)

    A = build_dense_block_matrix(D, L, U, N_BLOCKS)
    rhs = smooth_rhs(B, N_BLOCKS)
    x_dense = np.linalg.solve(A, rhs)
    x_round = block_schur_solve(L, U, rounded_inverse_blocks, rhs)
    x_greedy = block_schur_solve(L, U, greedy_inverse_blocks, rhs)

    full_summary = pd.DataFrame(
        [
            {
                "method": "dense_direct_reference",
                "relative_error_vs_dense": 0.0,
                "relative_residual": float(np.linalg.norm(A @ x_dense - rhs) / np.linalg.norm(rhs)),
                "max_local_residual": 0.0,
                "mean_local_residual": 0.0,
                "median_objective_improvement_fraction": "",
                "columns_improved_fraction": "",
                "runtime_seconds_for_greedy_polish": "",
            },
            {
                "method": "rounded_cached_inverse",
                "relative_error_vs_dense": float(np.linalg.norm(x_round - x_dense) / np.linalg.norm(x_dense)),
                "relative_residual": float(np.linalg.norm(A @ x_round - rhs) / np.linalg.norm(rhs)),
                "max_local_residual": float(full_columns["rounded_residual_norm"].max()),
                "mean_local_residual": float(full_columns["rounded_residual_norm"].mean()),
                "median_objective_improvement_fraction": 0.0,
                "columns_improved_fraction": 0.0,
                "runtime_seconds_for_greedy_polish": "",
            },
            {
                "method": "greedy_polished_cached_inverse",
                "relative_error_vs_dense": float(np.linalg.norm(x_greedy - x_dense) / np.linalg.norm(x_dense)),
                "relative_residual": float(np.linalg.norm(A @ x_greedy - rhs) / np.linalg.norm(rhs)),
                "max_local_residual": float(full_columns["greedy_residual_norm"].max()),
                "mean_local_residual": float(full_columns["greedy_residual_norm"].mean()),
                "median_objective_improvement_fraction": float(
                    full_columns["objective_improvement_fraction"].median()
                ),
                "columns_improved_fraction": float(
                    np.mean(full_columns["objective_improvement_fraction"] > 1e-15)
                ),
                # Runtime is included for provenance only; it is machine dependent.
                "runtime_seconds_for_greedy_polish": polish_seconds,
            },
        ]
    )
    full_summary.to_csv(output_dir / "full_cached_solve_summary.csv", index=False)

    metadata = {
        "elapsed_seconds": time.perf_counter() - start_all,
        "representative_instances": 32,
        "full_inverse_columns_polished": 960,
        "seeds": {
            "cold": "100000 + 1000*block_1based + column",
            "warm": "200000 + 1000*block_1based + column",
        },
        "interpretation": "No exact 96-bit enumeration or ground-state certificate is claimed.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return representative, summary, full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "b8_optimizer",
        help="Directory for regenerated CSV/JSON outputs.",
    )
    args = parser.parse_args()

    _, summary, full_summary = run_experiment(args.output_dir)
    print("Representative optimizer summary:\n")
    print(summary.to_string(index=False))
    print("\nFull cached-solve summary:\n")
    print(full_summary.to_string(index=False))
    print(f"\nSaved outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
