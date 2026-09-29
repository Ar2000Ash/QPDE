"""Solve QUBO matrices sequentially by exhaustive binary search."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class QUBOResult:
    bitstring: str
    energy: float
    n_bits: int
    states_evaluated: int
    elapsed_seconds: float
    backend: str


def solve_qubos(
    qubos: Iterable[np.ndarray] | np.ndarray,
    *,
    device: str = 'cuda',
    chunk_power: int = 16,
    prefix_batch: int = 32,
    max_bits: int = 24,
) -> list[QUBOResult]:
    """Minimize QUBOs sequentially and return their bitstrings and energies."""
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

            if low_bits != last_low_bits:
                codes = torch.arange(n_suffix, dtype=torch.int64, device=device)
                shifts = torch.arange(low_bits - 1, -1, -1, dtype=torch.int64, device=device)
                suffix_bits = ((codes[:, None] >> shifts[None, :]) & 1).to(torch.float64)
                last_low_bits = low_bits
            assert suffix_bits is not None
            q = torch.as_tensor(Q, dtype=torch.float64, device=device)
            A, B = q[:high_bits, :high_bits], q[:high_bits, high_bits:]
            C, D = q[high_bits:, :high_bits], q[high_bits:, high_bits:]
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
                local_index = torch.argmin(flat)
                candidate = flat[local_index]
                global_index = (prefix_codes[0] << low_bits) + local_index
                improve = candidate < best_energy
                best_index = torch.where(improve, global_index, best_index)
                best_energy = torch.minimum(best_energy, candidate)

            index = int(best_index.item())
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
    """Run the QUBO solver on the CUDA device."""
    if 'device' in kwargs:
        raise TypeError('solve_qubos_cuda fixes device=cuda; use solve_qubos for CPU tests')
    return solve_qubos(qubos, device='cuda', **kwargs)
