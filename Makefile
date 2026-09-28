PYTHON ?= python

.PHONY: reproduce five-pde sweep tables figures verify qci experiments cuda-test

five-pde:
	$(PYTHON) finite_bit/src/figure3_control.py
	$(PYTHON) finite_bit/src/figure3_multiscale.py

sweep:
	$(PYTHON) finite_bit/tests/check_figure2.py

tables: five-pde
	$(PYTHON) analysis/build_processed_data.py

figures: tables
	$(PYTHON) analysis/generate_figures.py

verify: tables
	$(PYTHON) analysis/verify_results.py
	$(PYTHON) finite_bit/tests/check_oracle.py
	$(PYTHON) finite_bit/tests/check_boundaries.py
	$(PYTHON) finite_bit/tests/check_figure2.py
	$(PYTHON) finite_bit/tests/check_figure3_multiscale.py
	$(PYTHON) finite_bit/tests/check_figure3_multiscale_outputs.py

qci: tables
	$(PYTHON) analysis/check_qci_input_qubos.py
	$(PYTHON) analysis/check_dirac3_decode.py
	$(PYTHON) analysis/reconstruct_dirac3_pde.py --write
	$(PYTHON) analysis/audit_dirac3.py
	$(PYTHON) analysis/verify_qci_recovery.py

experiments:
	$(PYTHON) experiments/run_large_block.py
	$(PYTHON) experiments/run_b8_optimizer.py
	$(PYTHON) analysis/check_experiment_outputs.py

cuda-test:
	$(PYTHON) finite_bit/tests/check_batch_qubo_cuda.py

reproduce: figures verify qci experiments cuda-test
