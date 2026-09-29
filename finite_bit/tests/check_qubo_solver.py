"""Check the QUBO solver against independently enumerated minima."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qubo_solver import solve_qubos


def reference(Q):
    n = len(Q)
    states = np.arange(1 << n, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.float64)
    scores = np.einsum('bi,ij,bj->b', bits, Q, bits, optimize=True)
    idx = int(np.argmin(scores))
    return format(idx, f'0{n}b'), float(bits[idx] @ Q @ bits[idx])


def run(device='cpu'):
    rng = np.random.default_rng(20260928)
    matrices = [np.zeros((4, 4)), np.diag([-2., 3., -1., 0.]),
                np.array([[0., -3.], [1., 2.]])]
    for n in (1, 2, 5, 8):
        for case in range(8):
            q = rng.normal(size=(n, n))
            if case % 2:
                q = np.triu(q)
            matrices.append(q)
    expected = [reference(q) for q in matrices]
    outputs = solve_qubos(matrices, device=device, chunk_power=3, prefix_batch=2, max_bits=12)
    assert len(outputs) == len(expected)
    for i, (actual, (bits, score)) in enumerate(zip(outputs, expected)):
        assert actual.bitstring == bits, (device, i, actual.bitstring, bits)
        np.testing.assert_allclose(actual.energy, score, atol=1e-10, rtol=1e-12)
        assert actual.states_evaluated == 1 << actual.n_bits
    stack = np.stack([np.eye(4), np.zeros((4, 4))])
    assert len(solve_qubos(stack, device=device, chunk_power=3, max_bits=12)) == 2
    assert len(solve_qubos(stack[0], device=device, chunk_power=3, max_bits=12)) == 1
    for bad in (np.array([[float('nan')]]), np.zeros((2, 3)), np.zeros((13, 13))):
        try:
            solve_qubos([bad], device=device, max_bits=12)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid QUBO was accepted')
    print(f'PASS: {device} sequential QUBO solver matches {len(expected)} independently enumerated minima')


if __name__ == '__main__':
    try:
        import torch
    except ImportError:
        print('SKIP: PyTorch is not installed')
        raise SystemExit(0)
    run('cpu')
