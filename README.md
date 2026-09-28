# Reusable QUBO-Derived Block-Schur Inverse Factors for Fixed-Operator PDE Solvers

Source code, numerical records, and plotting data accompanying the paper *Reusable QUBO-Derived Block-Schur Inverse Factors for Fixed-Operator Finite-Difference PDE Solvers*.

The offline calculation solves compact least-squares QUBOs for columns of local Schur inverses. These columns are decoded and cached. Each subsequent right-hand side is treated by ordinary classical block forward/backward substitution. The five-PDE validation uses four finite-grid optimizations per inverse column with the predetermined scale `gamma_p = gamma0 / 8**p` for `p = 0, 1, 2, 3`.

## Quick start

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce
```

`make reproduce` recomputes the finite-bit sweep and five-PDE experiment, regenerates processed data and figures, verifies the recorded QCI bitstrings and terminal fields, and runs the large-block and 96-bit studies. Results are written to `finite_bit/outputs/`, `outputs/`, `data/processed/`, and `figures/`. The original scientific records under `data/raw/` are not modified. Source-level checks also run in [GitHub Actions](.github/workflows/reproduce.yml).

The plotting script uses the paper's serif typography, with native LaTeX text when a LaTeX installation is present. Figure files, including the native PGFPlots source for Figure 3, are kept together under [`figures/`](figures/). The Figure 3 source reads `data/processed/figure3_multiscale_plot.csv`.

## Results and regeneration

| Paper item | Main source | Reproduction |
|:--|:--|:--|
| Figure 1, Table 1 | Mathematical description in paper | Architecture and definitions |
| Table 2 | `finite_bit/src/pde_models.py`, `finite_bit/src/operators.py` | Benchmark operators, forcing and conditions |
| Figure 2 | `finite_bit/src/figure2_exact_reconstruction.py` | Compute and check the fourteen precision-sweep rows |
| Table 3, Figure 3 | `finite_bit/src/figure3_multiscale.py` | Compute all five PDEs, field values and four-stage logs |
| Table 4, Figure 4 | `experiments/run_large_block.py` | Fixed-`N=960` Poisson block-size sweep |
| Table 5, Figure 5 | `experiments/run_b8_optimizer.py` | 96-bit heuristic optimization and complete cached solve |
| Table 6, Figures 6–7 | `analysis/reconstruct_dirac3_pde.py` | Re-decode recorded QCI bitstrings and rebuild terminal fields |

The central numerical checks are:

```bash
python finite_bit/tests/check_figure2.py
python finite_bit/tests/check_figure3_multiscale.py
python finite_bit/tests/check_figure3_multiscale_outputs.py
python analysis/check_qci_input_qubos.py
python analysis/reconstruct_dirac3_pde.py --write
python analysis/verify_results.py
```

The five-PDE experiment uses `B=2, M=1, K=10`, with initial scales 0.60 (Heat, Burgers), 0.01 (Poisson, Helmholtz), and 1.00 (Klein–Gordon). For the transient problems `dt=0.001` and `T=0.05`. Its 480 local minimizations are performed by the exact two-scalar reduced-grid CPU oracle. The reported `2**24` possible states per problem describe the logical problem size, not measured CPU/GPU work. A separate single-grid control is provided under `figure3_control.py`; the manuscript's main Figure 3 uses only the multiscale results.

## QCI-derived PDE fields

The stored QCI campaign comprises 34 distinct submitted QUBOs mapped to 60 inverse-column uses across Heat, Poisson and Klein–Gordon. The 24-bit solutions are decoded with fixed scales `gamma = 0.60, 0.01, 1.00`, respectively; two decoded columns form each 2×2 inverse block. The implementation verifies the saved Schur blocks, reconstructs the block-tridiagonal operator and applies its QCI-derived inverse factors by classical forward/backward substitution.

```bash
python analysis/verify_qci_recovery.py
python analysis/check_qci_input_qubos.py
python analysis/reconstruct_dirac3_pde.py --write
python analysis/build_processed_data.py
python analysis/generate_figures.py
```

The recorded dense terminal reference defines the reproducible right-hand side `b = A @ u_ref`. This verifies the complete terminal cached linear solve and its field values, not an unavailable full transient forcing history. The original QCI device response data are retained; the directly accessible QUBO coefficient tables reconstruct the same mathematical objectives but are not byte-identical copies of the original vendor upload files. Physical device execution requires external QCI access.

## CUDA QUBO utility

`finite_bit/src/batch_qubo_cuda.py` implements `solve_qubos_cuda(qubos)` for a sequence of QUBO matrices. It processes matrices one by one using bounded float64 CUDA buffers and exhaustive binary search; there is no silent CPU fallback. The optional dependency is a CUDA-enabled PyTorch installation appropriate for your driver. Small-instance CPU and optional GPU tests:

```bash
python finite_bit/tests/check_batch_qubo_cuda.py
```

This general-purpose GPU utility is not the measured backend for the five-PDE CPU results.

## Layout

```text
src/qpde_schur/       Core block-Schur and fixed-point operations
finite_bit/src/       Exact local QUBO oracle, five-PDE benchmarks and CUDA utility
finite_bit/tests/     Independent numerical and algorithm tests
finite_bit/outputs/   Five-PDE tables, fields and inverse-column logs
experiments/          Large-block and 96-bit experiments
analysis/             Hardware reconstruction, processing and plotting
data/raw/             Primary numerical and QCI records
data/reconstructed_qubos/  Directly accessible QCI mathematical inputs
data/provenance/      Recorded QCI device-response package and checksums
data/processed/       Publication figure inputs
figures/              Figures 2–7, Figure 3 PGFPlots source, and shared style
```

The requirements and scripts are independent of account credentials. Numerical reference arrays and recorded bitstrings are scientific fixtures; optimizer-quality and hardware measurements are reported separately from exact CPU and classical results.

## Figures

The six numerical figure previews are provided in [`figures/`](figures/). Regenerate the PNGs with `python analysis/build_processed_data.py` followed by `python analysis/generate_figures.py`. The native manuscript Figure 3 is [`figures/figure_03_five_pde_validation.tex`](figures/figure_03_five_pde_validation.tex); include [`figures/qpde_figure_style.tex`](figures/qpde_figure_style.tex) in the LaTeX preamble when using this source. Both the PNG generator and the PGFPlots file read the same `data/processed/figure3_multiscale_plot.csv`.

## Citation

Please cite the associated paper and the software package; author details are provided in [`CITATION.cff`](CITATION.cff).
