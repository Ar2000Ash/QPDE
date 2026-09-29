"""Block-Schur and QUBO numerical utilities."""

from .core import (
    block_schur_solve,
    build_dense_block_matrix,
    gamma_from_block,
    poisson_line_blocks,
    schur_inverse_factors,
    smooth_rhs,
)

__all__ = [
    "block_schur_solve",
    "build_dense_block_matrix",
    "gamma_from_block",
    "poisson_line_blocks",
    "schur_inverse_factors",
    "smooth_rhs",
]
