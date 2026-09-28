# QCI hardware record and terminal reconstruction

The saved QCI response bundle contains the returned bitstrings, submission metadata and job identifiers for 34 unique 24-bit QUBOs. Repeated local blocks map them to 60 inverse columns.

The file `data/raw/dirac3/reconstructed_60_instances.csv` contains each Schur matrix, column index, signed fixed-point scale and selected hardware bitstring. The 34 directly accessible coefficient tables in `data/reconstructed_qubos/` specify the mathematically reconstructed normalized input QUBOs. The original vendor-upload polynomial file bytes are not available, so their historical hashes must not be equated with reconstructed coefficients.

The complete saved response package is retained in the nine ordered parts of `data/provenance/qci_reconstruction_2026_09_26/parts/`. Run `python analysis/verify_qci_recovery.py` to verify and extract it in a temporary directory.

For each PDE, `analysis/reconstruct_dirac3_pde.py` independently decodes the two scalar values in every 24-bit bitstring, fills the two columns of every 2×2 inverse block, reconstructs and verifies the 20×20 block-tridiagonal operator, and applies the cached block forward/backward solve. The final right-hand side is defined as `A @ u_ref` from the saved dense terminal reference. This tests the reported terminal linear-system reconstruction; it does not claim to reproduce the original transient forcing history. No live QCI access is needed for offline verification.
