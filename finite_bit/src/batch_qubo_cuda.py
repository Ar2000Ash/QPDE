"""Sequential exhaustive solution of QUBO matrices with bounded device memory.

Each QUBO is processed independently. CUDA requires a compatible PyTorch build.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class QUBOResult:
    bitstring: str                    # most significant bit first
    energy: float                     # x.T @ Q @ x, recomputed in CPU float64
    n_bits: int
    states_evaluated: int             # logical states, not CUDA thread count
    elapsed_seconds: float            # includes matrix upload and device work
    backend: str


def solve_qubos(
    qubos: Iterable[np.ndarray] | np.ndarray,
    *,
    device: str = 'cuda',
    chunk_power: int = 16,
    prefix_batch: int = 32,
    max_bits: int = 24,
) -> list[QUBOResult]:
    """Solve an array/iterable of QUBOs one at a time using exhaustive search.

    Parameters
    ----------
    qubos : (jobs,n,n) ndarray or iterable of (n,n) matrices
        A 2-D ndarray is also accepted as one job; an iterable may have mixed
        matrix sizes. Input matrices stay on the host until their turn.
    device : {'cuda', 'cpu'}
        CUDA is the normal backend; CPU is useful for small verification cases.
    chunk_power : int
        Enumerate 2**min(chunk_power,n) reusable suffix states per job.
        Default 16 retains ~8 MiB of float64 suffix bit vectors for n>=16.
    prefix_batch : int
        Number of high-bit assignments evaluated together; controls the
        transient (prefix_batch, 2**chunk_power) score buffer.
    max_bits : int
        Explicit complexity guard. Default 24 allows 16,777,216 states/job;
        increasing it can make exhaustive search prohibitively expensive.

    Returns
    -------
    list[QUBOResult]
        Order matches the input. Equal objective scores choose the smallest
        integer bitstring (000... first), under the chosen float64 arithmetic.

    PyTorch is optional to import this module but required to run it. GPU
    execution needs a CUDA-enabled build and device; it does not silently fall
    back to CPU. Near-degenerate floating-point minima may require higher
    precision checks to establish a mathematical certificate.
    """
    if device not in ('cuda', 'cpu'):
        raise ValueError("device must be 'cuda' or 'cpu'")
    if not isinstance(chunk_power, int) or not 1 <= chunk_power <= 20:
        raise ValueError('chunk_power must be between 1 and 20')
    if not isinstance(prefix_batch, int) or prefix_batch < 1:
        raise ValueError('prefix_batch must be a positive integer')
    if not isinstance(max_bits, int) or not 1 <= max_bits <= 30:
        raise ValueError('max_bits must be between 1 and 30')

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is required; install a CUDA-enabled build for GPU use') from exc
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('a CUDA-capable GPU and CUDA-enabled PyTorch build are required')

    if isinstance(qubos, np.ndarray):
        if qubos.ndim == 2:
            jobs = iter((qubos,))
        elif qubos.ndim == 3:
            jobs = iter(qubos)
        else:
            raise ValueError('ndarray input must have shape (n,n) or (jobs,n,n)')
    else:
        jobs = iter(qubos)

    results: list[QUBOResult] = []
    last_low_bits = -1
    suffix_bits = None
    with torch.inference_mode():
        for job_number, raw in enumerate(jobs):
            Q = np.asarray(raw, dtype=np.float64)
            if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not 1 <= Q.shape[0] <= max_bits:
                raise ValueError(f'QUBO {job_number}: expected a square (n,n) matrix with 1 <= n <= {max_bits}')
            if not np.isfinite(Q).all():
                raise ValueError(f'QUBO {job_number}: coefficients must be finite')
            n = Q.shape[0]
            low_bits = min(n, chunk_power)
            high_bits = n - low_bits
            n_suffix = 1 << low_bits
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = perf_counter()

            # Reuse the binary suffix table across consecutive jobs of equal
            # low-bit size. Do not retain multiple tables on GPU.
            if low_bits != last_low_bits:
                codes = torch.arange(n_suffix, dtype=torch.int64, device=device)
                shifts = torch.arange(low_bits - 1, -1, -1, dtype=torch.int64, device=device)
                suffix_bits = ((codes[:, None] >> shifts[None, :]) & 1).to(torch.float64)
                last_low_bits = low_bits
            assert suffix_bits is not None
            q = torch.as_tensor(Q, dtype=torch.float64, device=device)
            A, B = q[:high_bits, :high_bits], q[:high_bits, high_bits:]
            C, D = q[high_bits:, :high_bits], q[high_bits:, high_bits:]
            # E(p,s) = p^T A p + s^T D s + p^T(B+C^T)s.
            low_energy = ((suffix_bits @ D) * suffix_bits).sum(dim=1)
            cross_matrix = B + C.T
            prefix_shifts = torch.arange(high_bits - 1, -1, -1, dtype=torch.int64, device=device)
            best_energy = torch.full((), float('inf'), dtype=torch.float64, device=device)
            best_index = torch.zeros((), dtype=torch.int64, device=device)

            for begin in range(0, 1 << high_bits, prefix_batch):
                end = min(begin + prefix_batch, 1 << high_bits)
                prefix_codes = torch.arange(begin, end, dtype=torch.int64, device=device)
                if high_bits:
                    prefix = ((prefix_codes[:, None] >> prefix_shifts[None, :]) & 1).to(torch.float64)
                    high_energy = ((prefix @ A) * prefix).sum(dim=1)
                    cross = prefix @ cross_matrix
                    scores = (cross @ suffix_bits.T) + high_energy[:, None] + low_energy[None, :]
                else:
                    scores = low_energy.unsqueeze(0)
                flat = scores.reshape(-1)
                local_index = torch.argmin(flat)  # lowest index wins a tie
                candidate = flat[local_index]
                global_index = (prefix_codes[0] << low_bits) + local_index
                improve = candidate < best_energy  # retain earlier index on ties
                best_index = torch.where(improve, global_index, best_index)
                best_energy = torch.minimum(best_energy, candidate)

            index = int(best_index.item())  # one result transfer per QUBO
            bits = format(index, f'0{n}b')
            binary = np.fromiter((int(b) for b in bits), dtype=np.float64, count=n)
            energy = float(binary @ Q @ binary)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = perf_counter() - start_time
            results.append(QUBOResult(bits, energy, n, 1 << n, elapsed,
                                      'cuda_exhaustive_float64' if device == 'cuda' else 'cpu_exhaustive_float64'))
    return results


def solve_qubos_cuda(qubos: Iterable[np.ndarray] | np.ndarray, **kwargs) -> list[QUBOResult]:
    """Convenience entry point that always selects CUDA (no CPU fallback)."""
    if 'device' in kwargs:
        raise TypeError('solve_qubos_cuda fixes device=cuda; use solve_qubos for CPU tests')
    return solve_qubos(qubos, device='cuda', **kwargs)
