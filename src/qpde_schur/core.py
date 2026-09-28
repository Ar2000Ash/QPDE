"""Numerical building blocks shared by the rerunnable experiments.

The study considers finite-difference systems that are ordered into block-tridiagonal
form.  The functions in this module deliberately keep that algebra explicit: they
build the line-block Poisson test matrix, form dense Schur inverse factors, apply a
cached block solve, and compute the fixed-point scale used in the larger-block tests.

These routines use NumPy only and contain no device- or vendor-specific code.
"""

from __future__ import annotations

import numpy as np


def poisson_line_blocks(block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three repeating blocks of a 5-point Poisson line ordering.

    Parameters
    ----------
    block_size:
        Number of grid unknowns stored in one line block.

    Returns
    -------
    D, L, U:
        Diagonal, lower-neighbor, and upper-neighbor blocks.  The matrix assembled
        from these blocks has diagonal entries 4 and nearest-neighbor entries -1.
    """
    b = int(block_size)
    if b < 1:
        raise ValueError("block_size must be positive")

    D = 4.0 * np.eye(b)
    for j in range(b - 1):
        D[j, j + 1] = -1.0
        D[j + 1, j] = -1.0
    L = -np.eye(b)
    U = -np.eye(b)
    return D, L, U


def build_dense_block_matrix(
    D: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """Assemble a dense block-tridiagonal matrix for validation purposes.

    The validation uses this dense matrix only as a reference on the modest test sizes.
    Production sparse solvers would of course avoid materializing the full matrix.
    """
    b = D.shape[0]
    n = b * int(n_blocks)
    A = np.zeros((n, n), dtype=float)

    for i in range(n_blocks):
        sl = slice(i * b, (i + 1) * b)
        A[sl, sl] = D
        if i < n_blocks - 1:
            nxt = slice((i + 1) * b, (i + 2) * b)
            A[sl, nxt] = U
            A[nxt, sl] = L
    return A


def schur_inverse_factors(
    D: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    n_blocks: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Form the Schur blocks and their dense inverse factors.

    This function represents the *classical reference* used in the large-block
    representability study and to seed the 96-bit heuristic experiment.  It does not
    imply that large QUBOs were solved exactly.
    """
    schur_blocks: list[np.ndarray] = []
    inverse_blocks: list[np.ndarray] = []

    S = D.copy()
    Y = np.linalg.inv(S)
    schur_blocks.append(S)
    inverse_blocks.append(Y)

    for _ in range(1, int(n_blocks)):
        S = D - L @ Y @ U
        Y = np.linalg.inv(S)
        schur_blocks.append(S)
        inverse_blocks.append(Y)

    return schur_blocks, inverse_blocks


def gamma_from_block(
    S: np.ndarray,
    S_inv: np.ndarray,
    integer_bits: int = 1,
) -> tuple[float, float, str]:
    """Compute the per-block fixed-point scale used in the reported experiments.

    For strictly diagonally dominant Schur blocks we use the analytical dominance
    margin.  If that condition fails, the infinity norm of the inverse provides a
    conservative fallback.  The returned mode string makes the choice auditable.
    """
    offdiag_sum = np.sum(np.abs(S), axis=1) - np.abs(np.diag(S))
    alpha = np.abs(np.diag(S)) - offdiag_sum
    alpha_min = float(np.min(alpha))

    if alpha_min > 0.0:
        gamma = (2.0 ** (-integer_bits)) / alpha_min
        return gamma, alpha_min, "diag_dominance"

    gamma = (2.0 ** (-integer_bits)) * float(np.linalg.norm(S_inv, ord=np.inf))
    return gamma, alpha_min, "inverse_norm_fallback"


def smooth_rhs(block_size: int, n_blocks: int) -> np.ndarray:
    """Return the smooth sine right-hand side used in the large-block studies."""
    xs = np.arange(1, n_blocks + 1, dtype=float) / (n_blocks + 1)
    ys = np.arange(1, block_size + 1, dtype=float) / (block_size + 1)
    return np.array(
        [np.sin(np.pi * x) * np.sin(np.pi * y) for x in xs for y in ys],
        dtype=float,
    )


def block_schur_solve(
    L: np.ndarray,
    U: np.ndarray,
    inverse_blocks: list[np.ndarray],
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a block-tridiagonal system using cached Schur inverse factors."""
    b = inverse_blocks[0].shape[0]
    n_blocks = len(inverse_blocks)

    # Forward elimination: build the modified right-hand-side blocks.
    g: list[np.ndarray] = [np.empty(b) for _ in range(n_blocks)]
    g[0] = rhs[:b].copy()
    for i in range(1, n_blocks):
        rhs_i = rhs[i * b : (i + 1) * b]
        g[i] = rhs_i - L @ inverse_blocks[i - 1] @ g[i - 1]

    # Back substitution using the same cached inverse factors.
    x: list[np.ndarray] = [np.empty(b) for _ in range(n_blocks)]
    x[-1] = inverse_blocks[-1] @ g[-1]
    for i in range(n_blocks - 2, -1, -1):
        x[i] = inverse_blocks[i] @ (g[i] - U @ x[i + 1])

    return np.concatenate(x)
