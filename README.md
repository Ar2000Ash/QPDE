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

The default workflow recomputes the five-PDE validation, builds the processed datasets and Figures 2–7, checks the archived hardware records, and runs the larger-block studies. The QCI checks use recorded device results; they do not submit new hardware jobs. A PyTorch installation is optional for the standalone QUBO solver.

## Experiments

| Paper item | Numerical source |
|---|---|
| Figure 2: finite-bit precision | `finite_bit/src/figure2_exact_reconstruction.py` |
| Table 3 and Figure 3: five-PDE validation | `finite_bit/src/figure3_multiscale.py` |
| Table 4 and Figure 4: Poisson block sizes | `experiments/run_large_block.py` |
| Table 5 and Figure 5: 96-bit optimization | `experiments/run_b8_optimizer.py` |
| Table 6 and Figures 6–7: QCI results | `analysis/reconstruct_dirac3_pde.py` |

The five-PDE validation uses `B=2`, `M=1`, `K=10` and four correction scales, `gamma_p = gamma0 / 8**p`, for `p=0,1,2,3`. Initial scales are 0.60 for Heat and Burgers, 0.01 for Poisson and Helmholtz, and 1.00 for Klein–Gordon. The transient cases use `dt=0.001` and `T=0.05`. The local minimizations use an exact reduced-grid finite-bit solver.

The five-PDE relative errors against the dense discrete solutions (Table 3) are 1.34e-5 (Heat), 1.10e-5 (Burgers), 3.05e-6 (Poisson), 6.88e-6 (Helmholtz), and 2.30e-5 (Klein–Gordon). Across the five cases, the experiment makes 480 inverse-column QUBO calls, including 360 nonzero correction updates. The fixed-grid control is archived in `finite_bit/outputs/figure3_control.csv`.

Table 4 and Figure 4 report **four** Poisson block sizes, `B = 4, 8, 16, 32`, for `N = 960`, `M = 1`, and `K = 10`. These correspond to 48, 96, 192, and 384 bits per inverse-column QUBO; the respective dense upper-triangular QUBO term counts are 1,176, 4,656, 18,528, and 73,920. The maximum local Schur condition number ranges from 2.96 to 5.29. The largest reported relative solution error is 2.35e-3, and the `B = 32` relative residual is 1.20e-1. The original `B = 2` diagnostic is retained in `data/raw/large_block/` and `data/processed/largeB.csv` for reproducibility but is **not** part of Table 4 or Figure 4.

The `B = 8` study (Table 5) compares direct fixed-point rounding, greedy descent, and cold-/warm-start simulated annealing on 32 representative 96-bit QUBOs. Cold and warm annealing use 140 and 120 sweeps **per restart**, respectively, with two restarts and one deterministic seed per method and instance, followed by at most eight greedy sweeps. These are optimization-quality tests, not claims of global solutions to 96-bit QUBOs.

The Dirac-3 campaign (Table 6) comprises 34 unique completed QUBO jobs mapped to 60 inverse-column uses, with 31/60 exact mapped bitstring hits, 17/34 exact unique-QUBO hits, and 558 s of metered device usage. Jobs used `sample-hamiltonian-integer`, two levels per variable, no sum constraint, and normalized constant-free coefficients. Of the 34 jobs, 32 requested 20 samples at relaxation schedule 1; two Poisson jobs requested 100 samples at schedule 4. The hardware study decodes the recorded bitstrings into cached Schur-inverse columns, reconstructs the terminal linear systems, and compares the resulting fields with dense references. The terminal right-hand sides are defined by `b = A @ u_ref`; the offline replay does not re-run the QCI hardware.

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
