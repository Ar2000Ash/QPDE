"""Verify the decoded QCI bitstrings and inverse-column metrics."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpde_schur.binary import decode_bits, make_bit_layout

RAW = ROOT / "data" / "raw" / "dirac3"
PROCESSED = ROOT / "data" / "processed"
LAYOUT = make_bit_layout(2, 1, 10)
TOL = 1e-9


def equal_numeric(actual: object, expected: float, label: str) -> None:
    """Use absolute tolerance appropriate for the stored hardware audit precision."""
    try:
        actual_value = float(actual)
    except (ValueError, TypeError) as exc:
        raise AssertionError(f"{label}: cannot parse {actual!r}") from exc
    if not math.isfinite(actual_value) or not math.isclose(
        actual_value, float(expected), rel_tol=1e-9, abs_tol=TOL
    ):
        raise AssertionError(f"{label}: stored={actual_value!r}, recomputed={expected!r}")


def independent_evaluation(row: pd.Series, bitstring: str) -> tuple[np.ndarray, float, float]:
    """Return decoded vector, least-squares residual, constant-free submitted energy."""
    if not bitstring or len(bitstring) > 24 or any(c not in "01" for c in bitstring):
        raise AssertionError(f"{row.qci_id}: invalid binary string {bitstring!r}")
    bitstring = bitstring.zfill(24)
    bits = np.fromiter((int(c) for c in bitstring), dtype=np.int8, count=24)
    y = decode_bits(bits, float(row.gamma), LAYOUT)
    S = np.array([[row.S00, row.S01], [row.S10, row.S11]], dtype=float)
    target = np.eye(2)[:, int(row.column)]
    local_residual = float(np.linalg.norm(S @ y - target, 2))
    submitted_energy = (local_residual**2 - 1.0) / float(row.coeff_scale)
    return y, local_residual, submitted_energy


def check_archive() -> None:
    reconstructed = pd.read_csv(RAW / "reconstructed_60_instances.csv", dtype={"global_min_bitstring": str, "hw_best_bitstring": str, "hw_exact_bitstring": str})
    samples = pd.read_csv(RAW / "returned_samples.csv", dtype={"bitstring": str})
    unique = pd.read_csv(RAW / "unique_jobs.csv", dtype={"best_bitstring": str, "exact_bitstring": str})
    mapped = pd.read_csv(RAW / "mapped_60_instances.csv", dtype={"dirac_bitstring": str, "exact_bitstring": str})
    processed = pd.read_csv(PROCESSED / "dirac_mapped.csv")

    assert len(reconstructed) == len(mapped) == len(processed) == 60
    assert len(unique) == 34
    assert len(samples) == 457

    by_qci = {row.qci_id: row for _, row in reconstructed.iterrows()}
    by_unique = {}
    for _, row in reconstructed.iterrows():
        by_unique.setdefault(row.unique_id, row)

    for _, row in reconstructed.iterrows():
        for prefix in ("hw_best", "hw_exact"):
            y, residual, energy = independent_evaluation(row, str(row[prefix + "_bitstring"]))
            for idx in range(2):
                equal_numeric(row[prefix + f"_decoded_y{idx}"], y[idx],
                              row.qci_id + "." + prefix + f".decoded_y{idx}")
            equal_numeric(row[prefix + "_inverse_residual"], residual,
                          row.qci_id + "." + prefix + ".inverse_residual")
            energy_column = (prefix + "_computed_submitted_energy"
                             if prefix == "hw_best" else "hw_exact_submitted_energy_no_constant")
            equal_numeric(row[energy_column], energy, row.qci_id + "." + energy_column)

        exact_y, exact_residual, _ = independent_evaluation(row, str(row.global_min_bitstring))
        for idx in range(2):
            equal_numeric(row[f"decoded_y{idx}"], exact_y[idx],
                          row.qci_id + f".original_decoded_y{idx}")
        equal_numeric(row.inverse_residual, exact_residual,
                      row.qci_id + ".original_inverse_residual")

    for _, sample in samples.iterrows():
        ref = by_unique[sample.unique_id]
        y, residual, energy = independent_evaluation(ref, str(sample.bitstring))
        for idx in range(2):
            equal_numeric(sample[f"decoded_y{idx}"], y[idx],
                          sample.unique_id + f".sample_{sample.sample_index}.decoded_y{idx}")
        equal_numeric(sample.inverse_residual, residual,
                      sample.unique_id + f".sample_{sample.sample_index}.residual")
        equal_numeric(sample.computed_submitted_energy, energy,
                      sample.unique_id + f".sample_{sample.sample_index}.energy")

    for _, job in unique.iterrows():
        ref = by_unique[job.unique_id]
        for prefix, bitkey, energykey, reskey in (
            ("best", "best_bitstring", "best_computed_submitted_energy", "best_inverse_residual"),
            ("exact", "exact_bitstring", "exact_submitted_energy_no_constant", "exact_inverse_residual"),
        ):
            y, residual, energy = independent_evaluation(ref, str(job[bitkey]))
            for idx in range(2):
                equal_numeric(job[prefix + f"_decoded_y{idx}"], y[idx],
                              job.unique_id + "." + prefix + f".decoded_y{idx}")
            equal_numeric(job[reskey], residual, job.unique_id + "." + reskey)
            equal_numeric(job[energykey], energy, job.unique_id + "." + energykey)

    for _, row in mapped.iterrows():
        ref = by_qci[row.qci_id]
        y, residual, energy = independent_evaluation(ref, str(row.dirac_bitstring))
        equal_numeric(row.local_inverse_residual_dirac, residual,
                      row.qci_id + ".mapped_hardware_residual")
        equal_numeric(row.dirac_objective, energy, row.qci_id + ".mapped_energy")
        exact_y, exact_residual, exact_energy = independent_evaluation(
            ref, str(row.exact_bitstring)
        )
        equal_numeric(row.local_inverse_residual_exact, exact_residual,
                      row.qci_id + ".mapped_exact_residual")
        equal_numeric(row.exact_ground_objective, exact_energy,
                      row.qci_id + ".mapped_exact_energy")

    for _, row in processed.iterrows():
        ref = by_qci[row.qci_id]
        for short, full in (("hw", "hw_best"), ("exact", "hw_exact")):
            y, residual, _ = independent_evaluation(ref, str(ref[full + "_bitstring"]))
            for idx in range(2):
                equal_numeric(row[short + f"_y{idx}"], y[idx],
                              row.qci_id + "." + short + f"_y{idx}")
            equal_numeric(row[short + "_resid"], residual,
                          row.qci_id + "." + short + "_resid")

    assert int(mapped["exact_ground_state_match"].sum()) == 31
    assert int(unique["strict_exact_hit_any"].sum()) == 17
    assert int(unique["device_usage_s"].sum()) == 558

    print("Signed-decoding audit passed: 457 samples, 34 unique QUBOs, "
          "60 reconstructed and processed columns.")


if __name__ == "__main__":
    check_archive()
