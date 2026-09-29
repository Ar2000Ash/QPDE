# QPDE

**Reusable QUBO-Derived Block-Schur Inverse Factors for Fixed-Operator Finite-Difference PDE Solvers**

Numerical code and data for the precision sweep, five-PDE validation, large-block Poisson studies, and recorded QCI experiments.

## Setup

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce
```

The default workflow rebuilds the five-PDE results, verifies all numerical tables, regenerates Figures 2–7, checks the recorded QCI results, and runs the larger-block experiments. A PyTorch installation is optional for the standalone QUBO solver.

## Experiments

| Paper item | Numerical source |
|---|---|
| Figure 2: finite-bit precision | `finite_bit/src/figure2_exact_reconstruction.py` |
| Table 3 and Figure 3: five-PDE validation | `finite_bit/src/figure3_multiscale.py` |
| Table 4 and Figure 4: Poisson block sizes | `experiments/run_large_block.py` |
| Table 5 and Figure 5: 96-bit optimization | `experiments/run_b8_optimizer.py` |
| Table 6 and Figures 6–7: QCI results | `analysis/reconstruct_dirac3_pde.py` |

The five-PDE validation uses `B=2`, `M=1`, `K=10` and four correction scales, `gamma_p = gamma0 / 8**p`, for `p=0,1,2,3`. Initial scales are 0.60 for Heat and Burgers, 0.01 for Poisson and Helmholtz, and 1.00 for Klein–Gordon. The transient cases use `dt=0.001` and `T=0.05`. The local minimizations use an exact reduced-grid finite-bit solver.

The hardware study decodes the recorded QCI bitstrings into cached Schur-inverse columns, reconstructs the terminal linear systems, and compares the resulting fields with the dense references. The terminal right-hand sides are defined by `b = A @ u_ref`.

## Directory layout

```text
src/qpde_schur/       Block-Schur and fixed-point algorithms
finite_bit/src/       PDE benchmarks and finite-bit solvers
finite_bit/tests/     Numerical checks
finite_bit/outputs/   Five-PDE results and per-stage records
experiments/          Block-size and 96-bit studies
analysis/             Verification, data preparation, and plotting
data/raw/             Numerical and QCI records
data/reconstructed_qubos/  QCI input coefficient tables
data/provenance/      Device-response records and checksums
data/processed/       Table and figure inputs
figures/              Figures 2–7 and native Figure 3 LaTeX source
```

Use `python analysis/build_processed_data.py` and `python analysis/generate_figures.py` to regenerate the figure data and plots. Figure 3's PGFPlots source is `figures/figure_03_five_pde_validation.tex`, with input `data/processed/figure3_multiscale_plot.csv`.

The standalone QUBO solver is `finite_bit/src/qubo_solver.py`. Its small-instance verification is `finite_bit/tests/check_qubo_solver.py`.

See [`CITATION.cff`](CITATION.cff) for citation details.
