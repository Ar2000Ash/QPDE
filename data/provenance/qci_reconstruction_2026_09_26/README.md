# QCI response data

The nine files in `parts/` contain the recorded device responses for 34 distinct QUBOs, mapped to 60 inverse-column uses. Verify the package with:

```bash
python analysis/verify_qci_recovery.py
```

The decoded bitstrings and Schur matrices are in `data/raw/dirac3/`. The normalized coefficient tables are in `data/reconstructed_qubos/`. The terminal field reconstruction is implemented in `analysis/reconstruct_dirac3_pde.py`, using the recorded dense reference to define `b = A @ u_ref`.
