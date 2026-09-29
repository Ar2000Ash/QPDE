"""Evaluate fixed-point representability across Poisson block sizes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpde_schur.binary import quantize_matrix
from qpde_schur.core import (
    block_schur_solve,
    build_dense_block_matrix,
    gamma_from_block,
    poisson_line_blocks,
    schur_inverse_factors,
    smooth_rhs,
)


DEFAULT_B_VALUES = (2, 4, 8, 16, 32)


def run_experiment(
    *,
    n_unknowns: int = 960,
    block_sizes: tuple[int, ...] = DEFAULT_B_VALUES,
    integer_bits: int = 1,
    fractional_bits: int = 10,
) -> pd.DataFrame:
    """Return one row of diagnostics for each requested block size."""
    rows: list[dict[str, object]] = []
    bits_per_scalar = 1 + integer_bits + fractional_bits

    for block_size in block_sizes:
        if n_unknowns % block_size != 0:
            raise ValueError(f"N={n_unknowns} is not divisible by B={block_size}")

        n_blocks = n_unknowns // block_size
        D, L, U = poisson_line_blocks(block_size)

        start = time.perf_counter()
        schur_blocks, exact_inverse_blocks = schur_inverse_factors(D, L, U, n_blocks)
        schur_time = time.perf_counter() - start

        quantized_inverse_blocks: list[np.ndarray] = []
        gammas: list[float] = []
        dominance_margins: list[float] = []
        scaling_modes: list[str] = []
        inverse_residuals: list[float] = []
        saturated_entries = 0
        total_entries = 0
        max_inverse_entry = 0.0

        for S, S_inv in zip(schur_blocks, exact_inverse_blocks):
            gamma, alpha_min, mode = gamma_from_block(S, S_inv, integer_bits)
            S_inv_q, saturated = quantize_matrix(
                S_inv,
                gamma,
                integer_bits=integer_bits,
                fractional_bits=fractional_bits,
            )

            quantized_inverse_blocks.append(S_inv_q)
            gammas.append(gamma)
            dominance_margins.append(alpha_min)
            scaling_modes.append(mode)
            saturated_entries += int(np.sum(saturated))
            total_entries += int(saturated.size)
            inverse_residuals.append(
                float(np.linalg.norm(S @ S_inv_q - np.eye(block_size), ord="fro"))
            )
            max_inverse_entry = max(max_inverse_entry, float(np.max(np.abs(S_inv))))

        rhs = smooth_rhs(block_size, n_blocks)
        A = build_dense_block_matrix(D, L, U, n_blocks)
        x_dense = np.linalg.solve(A, rhs)
        x_quantized = block_schur_solve(L, U, quantized_inverse_blocks, rhs)

        rel_error = float(np.linalg.norm(x_quantized - x_dense) / np.linalg.norm(x_dense))
        rel_residual = float(np.linalg.norm(A @ x_quantized - rhs) / np.linalg.norm(rhs))
        max_condition = max(float(np.linalg.cond(S, 2)) for S in schur_blocks)

        encoded_bits = block_size * bits_per_scalar
        qubo_terms = encoded_bits * (encoded_bits + 1) // 2

        rows.append(
            {
                "B": block_size,
                "N": n_unknowns,
                "line_blocks": n_blocks,
                "M": integer_bits,
                "K": fractional_bits,
                "bits_per_scalar": bits_per_scalar,
                "encoded_bits_per_inverse_column": encoded_bits,
                "upper_triangular_qubo_terms": qubo_terms,
                "qubo_density_rho_Q": 1.0,
                "dense_inverse_flops_2over3B3": (2.0 / 3.0) * block_size**3,
                "relative_error_vs_dense": rel_error,
                "relative_residual": rel_residual,
                "saturation_fraction": saturated_entries / total_entries,
                "max_schur_kappa_2": max_condition,
                "min_diagonal_dominance_margin": min(dominance_margins),
                "mean_gamma_i": float(np.mean(gammas)),
                "max_gamma_i": float(np.max(gammas)),
                "max_exact_inverse_entry_abs": max_inverse_entry,
                "max_inverse_frobenius_residual": max(inverse_residuals),
                "mean_inverse_frobenius_residual": float(np.mean(inverse_residuals)),
                "scaling_modes_used": ";".join(sorted(set(scaling_modes))),
                "schur_factor_generation_time_s": schur_time,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "large_block" / "summary.csv",
        help="CSV path for regenerated diagnostics.",
    )
    args = parser.parse_args()

    result = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
