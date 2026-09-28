"""Signed fixed-point encoding and small QUBO-space optimization utilities.

The 96-bit experiment compares a rounded fixed-point state, greedy single-bit
polishing, and simulated annealing followed by greedy polishing.  This module contains
exactly those encoding and local-search operations in a vendor-neutral form.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BitLayout:
    """Describe the signed fixed-point bits associated with each vector entry."""

    block_size: int
    integer_bits: int
    fractional_bits: int
    scalar_index: np.ndarray
    weights: np.ndarray

    @property
    def bits_per_scalar(self) -> int:
        return 1 + self.integer_bits + self.fractional_bits

    @property
    def n_bits(self) -> int:
        return self.block_size * self.bits_per_scalar


def make_bit_layout(block_size: int, integer_bits: int, fractional_bits: int) -> BitLayout:
    """Create the bit-to-scalar mapping for the signed fixed-point encoding.

    The released 96-bit experiment uses one sign bit, one integer bit, and ten
    fractional bits per scalar.  The implementation below also supports more than one
    integer bit so the representation is explicit rather than hard-coded to M=1.
    """
    scalar_index: list[int] = []
    weights: list[float] = []

    for ell in range(block_size):
        # Sign bit carries weight -2^M.
        scalar_index.append(ell)
        weights.append(-(2.0**integer_bits))

        # Non-negative integer bits have weights 2^0, ..., 2^(M-1).
        for m in range(integer_bits):
            scalar_index.append(ell)
            weights.append(2.0**m)

        # Fractional bits have weights 2^-1, ..., 2^-K.
        for k in range(1, fractional_bits + 1):
            scalar_index.append(ell)
            weights.append(2.0 ** (-k))

    return BitLayout(
        block_size=block_size,
        integer_bits=integer_bits,
        fractional_bits=fractional_bits,
        scalar_index=np.asarray(scalar_index, dtype=int),
        weights=np.asarray(weights, dtype=float),
    )


def integer_code_to_bits(code: int, layout: BitLayout) -> np.ndarray:
    """Encode one scaled integer value using the signed fixed-point convention."""
    m = layout.integer_bits
    k = layout.fractional_bits
    q = np.zeros(layout.bits_per_scalar, dtype=np.int8)

    offset = (2**m) * (2**k)
    if code < 0:
        q[0] = 1
        remainder = int(code + offset)
    else:
        remainder = int(code)

    # Integer bits occupy positions 1 ... M.
    integer_part, fractional_part = divmod(remainder, 2**k)
    for bit in range(m):
        q[1 + bit] = (integer_part >> bit) & 1

    # Fractional bits are stored from largest to smallest fractional weight.
    for frac_bit in range(1, k + 1):
        bit_value = 2 ** (k - frac_bit)
        if fractional_part >= bit_value:
            q[1 + m + frac_bit - 1] = 1
            fractional_part -= bit_value

    return q


def quantize_vector_to_bits(
    y: np.ndarray,
    gamma: float,
    layout: BitLayout,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Round a vector to the signed fixed-point grid and return its binary encoding."""
    m = layout.integer_bits
    k = layout.fractional_bits
    code_min = -(2**m) * (2**k)
    code_max = (2**m) * (2**k) - 1
    scale = (2**k) / gamma

    q = np.zeros(layout.n_bits, dtype=np.int8)
    y_quantized = np.zeros(layout.block_size, dtype=float)
    saturation_count = 0

    for ell in range(layout.block_size):
        raw_code = int(np.rint(float(y[ell]) * scale))
        code = max(code_min, min(code_max, raw_code))
        saturation_count += int(code != raw_code)

        start = ell * layout.bits_per_scalar
        stop = start + layout.bits_per_scalar
        q[start:stop] = integer_code_to_bits(code, layout)
        y_quantized[ell] = gamma * code / (2**k)

    return q, y_quantized, saturation_count


def quantize_matrix(
    matrix: np.ndarray,
    gamma: float,
    integer_bits: int,
    fractional_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Round a matrix directly to the same representable fixed-point grid."""
    code_min = -(2**integer_bits) * (2**fractional_bits)
    code_max = (2**integer_bits) * (2**fractional_bits) - 1
    scale = (2**fractional_bits) / gamma

    raw = np.rint(matrix * scale).astype(int)
    saturated = (raw < code_min) | (raw > code_max)
    clipped = np.clip(raw, code_min, code_max)
    quantized = clipped * gamma / (2**fractional_bits)
    return quantized, saturated


def decode_bits(q: np.ndarray, gamma: float, layout: BitLayout) -> np.ndarray:
    """Decode a binary vector back to the represented real vector."""
    coeff = gamma * layout.weights
    y = np.zeros(layout.block_size, dtype=float)
    np.add.at(y, layout.scalar_index, coeff * q)
    return y


def residual_energy(S: np.ndarray, column: int, q: np.ndarray, gamma: float, layout: BitLayout) -> float:
    """Return ||S y(q) - e_column||_2^2, the inverse-column QUBO objective."""
    e = np.zeros(layout.block_size, dtype=float)
    e[column] = 1.0
    residual = S @ decode_bits(q, gamma, layout) - e
    return float(residual @ residual)


def greedy_descent(
    S: np.ndarray,
    column: int,
    q0: np.ndarray,
    gamma: float,
    layout: BitLayout,
    max_sweeps: int = 8,
) -> tuple[np.ndarray, float, int, int]:
    """Best-improvement single-bit descent used in the reported experiment."""
    q = q0.copy()
    e = np.zeros(layout.block_size, dtype=float)
    e[column] = 1.0

    y = decode_bits(q, gamma, layout)
    residual = S @ y - e
    energy = float(residual @ residual)

    coeffs = gamma * layout.weights
    Scols = S[:, layout.scalar_index]
    colnorm2 = np.sum(Scols * Scols, axis=0)

    flips = 0
    sweeps_done = 0
    for _ in range(max_sweeps):
        improved = False
        sweeps_done += 1

        signs = 1 - 2 * q
        deltas = coeffs * signs
        delta_e = 2.0 * deltas * (Scols.T @ residual) + (deltas * deltas) * colnorm2
        k = int(np.argmin(delta_e))

        while delta_e[k] < -1e-18:
            delta = deltas[k]
            q[k] = 1 - q[k]
            residual = residual + delta * Scols[:, k]
            energy += float(delta_e[k])
            flips += 1
            improved = True

            signs = 1 - 2 * q
            deltas = coeffs * signs
            delta_e = 2.0 * deltas * (Scols.T @ residual) + (deltas * deltas) * colnorm2
            k = int(np.argmin(delta_e))

        if not improved:
            break

    return q, max(energy, 0.0), flips, sweeps_done


def simulated_annealing(
    S: np.ndarray,
    column: int,
    gamma: float,
    layout: BitLayout,
    *,
    q_start: np.ndarray | None = None,
    sweeps: int = 700,
    restarts: int = 3,
    seed: int = 0,
    warm_perturb: float = 0.0,
) -> tuple[np.ndarray, float, int, int]:
    """Run the deterministic-seed annealing protocol used for the 96-bit study."""
    rng = np.random.default_rng(seed)
    e = np.zeros(layout.block_size, dtype=float)
    e[column] = 1.0

    coeffs = gamma * layout.weights
    Scols = S[:, layout.scalar_index]
    colnorm2 = np.sum(Scols * Scols, axis=0)

    if q_start is not None:
        best_q = q_start.copy()
        best_energy = residual_energy(S, column, best_q, gamma, layout)
    else:
        best_q = None
        best_energy = float("inf")

    total_accepts = 0
    for _ in range(restarts):
        if q_start is None:
            q = rng.integers(0, 2, size=layout.n_bits, dtype=np.int8)
        else:
            q = q_start.copy()
            if warm_perturb > 0.0:
                mask = rng.random(layout.n_bits) < warm_perturb
                q[mask] = 1 - q[mask]

        residual = S @ decode_bits(q, gamma, layout) - e
        energy = float(residual @ residual)
        if energy < best_energy:
            best_energy = energy
            best_q = q.copy()

        t0 = max(0.05 * energy, 1e-3)
        t1 = 1e-8
        accepts = 0
        attempts = sweeps * layout.n_bits

        for step in range(attempts):
            frac = step / max(1, attempts - 1)
            temperature = t0 * ((t1 / t0) ** frac)
            bit = int(rng.integers(0, layout.n_bits))

            delta = coeffs[bit] * (1 - 2 * q[bit])
            delta_e = (
                2.0 * delta * float(Scols[:, bit] @ residual)
                + delta * delta * float(colnorm2[bit])
            )

            if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / max(temperature, 1e-300)):
                q[bit] = 1 - q[bit]
                residual = residual + delta * Scols[:, bit]
                energy += delta_e
                accepts += 1
                if energy < best_energy:
                    best_energy = float(energy)
                    best_q = q.copy()

        total_accepts += accepts

    if best_q is None:
        raise RuntimeError("annealing produced no candidate state")

    # The reported experiment uses annealing followed by the same greedy local polish.
    best_q, best_energy, greedy_flips, _ = greedy_descent(
        S,
        column,
        best_q,
        gamma,
        layout,
        max_sweeps=8,
    )
    return best_q, best_energy, total_accepts, greedy_flips
