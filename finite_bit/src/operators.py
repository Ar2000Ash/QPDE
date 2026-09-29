"""Assemble the five-PDE benchmark operators."""
from __future__ import annotations
from pde_models import elliptic_blocks, heat_matrix, kg_matrix


def fixed_operator(name: str):
    """Return the fixed matrix and, for elliptic systems, reference/RHS data."""
    if name == "heat_1d":
        A, *_ = heat_matrix(20, nu=0.08)
        return A, None
    if name == "burgers_1d":
        A, *_ = heat_matrix(20, nu=0.04)
        return A, None
    if name == "klein_gordon_1d":
        A, *_ = kg_matrix(20, 0.001, 1.1, 2.0)
        return A, None
    if name in ("poisson_2d", "helmholtz_2d"):
        _, _, _, A, exact, rhs, boundary = elliptic_blocks(name, 10, 2)
        return A, (exact, rhs, boundary)
    raise KeyError(name)
