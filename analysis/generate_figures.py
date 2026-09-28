#!/usr/bin/env python3
"""Regenerate the publication-quality result figures from processed CSV files.

The plotting layer is intentionally independent of the manuscript source.  Every panel
is generated from files under ``data/processed`` using Matplotlib.  When a LaTeX
installation is available, labels are rendered through LaTeX so that the typography
matches the figures used in the associated study.  A STIX serif fallback is used when
LaTeX is unavailable.

Outputs
-------
figures/figure_02_quantization_convergence.png
figures/figure_03_five_pde_validation.png
figures/figure_04_large_block_scaling.png
figures/figure_05_b8_optimizer.png
figures/figure_06_dirac3_audit.png
figures/figure_07_dirac3_reconstructions.png
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# A restrained print-safe palette.  Keeping it here makes all exported figures consistent.
BLUE = "#1f4e79"
TEAL = "#2a7f79"
ORANGE = "#c97a1d"
RED = "#a13d3d"
GRAY = "#555555"


def configure_matplotlib() -> None:
    """Apply a journal-style typographic configuration."""
    use_tex = shutil.which("latex") is not None
    mpl.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.dpi": 120,
        }
    )


def finish(fig: plt.Figure, filename: str) -> None:
    """Tighten and save one figure, then close it to release memory."""
    fig.tight_layout(pad=1.2)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_02() -> None:
    """Quantization-controlled convergence for Poisson and Klein--Gordon."""
    d = pd.read_csv(DATA / "k_sweep_compact.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)

    panels = [
        (axes[0], "Poisson 2D", "poisson", r"$s_u=-0.997,\ s_{S^{-1}}=-1.035,\ s_E=-1.932$"),
        (axes[1], "Klein--Gordon 1D", "kg", r"$s_u=-1.051,\ s_{S^{-1}}=-1.080,\ s_E=-1.940$"),
    ]
    for ax, title, prefix, slopes in panels:
        ax.semilogy(d["K"], d[f"{prefix}_solution"], "o-", color=BLUE, label="solution error")
        ax.semilogy(d["K"], d[f"{prefix}_inverse"], "s-", color=TEAL, label="mean inverse error")
        ax.semilogy(d["K"], d[f"{prefix}_energy"], "^-", color=ORANGE, label="mean QUBO energy")
        ax.set_title(title)
        ax.set_xlabel(r"Fractional bits $K$")
        ax.set_xticks(d["K"])
        ax.grid(True, which="both", alpha=0.18)
        ax.text(
            0.04,
            0.06,
            slopes,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            fontsize=8,
        )
    axes[0].set_ylabel("Measured quantity")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_ylim(1e-7, 2e-1)
    axes[1].set_ylim(1e-7, 7e-1)
    finish(fig, "figure_02_quantization_convergence.png")


def figure_03() -> None:
    """Five-PDE validation: final-state error and cached inverse residual."""
    d = pd.read_csv(DATA / "figure3_multiscale_plot.csv")
    assert d["short"].tolist() == ["Heat", "Burgers", "Poisson", "Helmholtz", "Klein--Gordon"]
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45))

    axes[0].semilogy(x, d["rel_dense"], "o", color=BLUE, ms=6)
    axes[0].set_title("(a) End-to-end relative error")
    axes[0].set_ylabel(r"$e_{\rm dense}$")
    axes[0].set_ylim(1e-7, 7e-5)

    axes[1].semilogy(x, d["max_inv_res"], "s", color=TEAL, ms=6)
    axes[1].set_title("(b) Maximum cached inverse residual")
    axes[1].set_ylabel(r"$\rho_S$")
    axes[1].set_ylim(1e-7, 1e-5)

    for ax in axes:
        ax.set_xlim(-0.45, len(x) - 0.55)
        ax.set_xticks(x, d["short"].tolist(), rotation=24, ha="right")
        ax.grid(True, which="both", alpha=0.18)
    finish(fig, "figure_03_five_pde_validation.png")


def figure_04() -> None:
    """Resource growth and propagated accuracy for the large-block study."""
    d = pd.read_csv(DATA / "largeB.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45))

    ax = axes[0]
    ax.loglog(d["B"], d["encoded_bits_per_inverse_column"], "o-", color=BLUE, label="encoded bits", base=2)
    ax.loglog(d["B"], d["upper_triangular_qubo_terms"], "s-", color=TEAL, label="QUBO terms", base=2)
    ax.loglog(d["B"], d["dense_inverse_flops_2over3B3"], "^-", color=ORANGE, label=r"$(2/3)B^3$", base=2)
    ax.set_title("(a) Local resource growth")
    ax.set_ylabel("Count / leading work")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(3, 1.2e5)

    ax = axes[1]
    # The B=2 solution error is near machine precision and intentionally lies below
    # the visible plotting range, matching the interpretation of the reported figure.
    ax.loglog(d["B"], d["relative_error_vs_dense"], "o-", color=BLUE, label="solution error", base=2)
    ax.loglog(d["B"], d["relative_residual"], "D-", color=RED, label="PDE residual", base=2)
    ax.set_title("(b) Propagated accuracy")
    ax.set_ylabel("Relative metric")
    ax.set_ylim(1e-4, 3e-1)
    ax.legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.set_xlabel(r"Block size $B$")
        ax.set_xticks(d["B"], [str(v) for v in d["B"]])
        ax.grid(True, which="both", alpha=0.18)
    finish(fig, "figure_04_large_block_scaling.png")


def figure_05() -> None:
    """Representative 96-bit QUBO optimization and full-factor downstream effect."""
    d = pd.read_csv(DATA / "b8_optimizer_plot.csv")
    full = pd.read_csv(DATA / "b8_full_solve.csv").set_index("method")
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))

    axes[0].semilogy(x, d["median_objective_ratio_to_rounded"], "o", color=BLUE, ms=6)
    axes[0].set_title("(a) Objective ratio")
    axes[0].set_ylabel("Median ratio to rounded")
    axes[0].set_ylim(5e-1, 1e5)

    axes[1].semilogy(x, d["median_residual_norm"], "s", color=TEAL, ms=6)
    axes[1].set_title("(b) Local residual")
    axes[1].set_ylabel(r"Median $r_S$")
    axes[1].set_ylim(5e-4, 1)

    methods = ["rounded_cached_inverse", "greedy_polished_cached_inverse"]
    x2 = np.arange(2)
    axes[2].semilogy(x2, [full.loc[m, "relative_error_vs_dense"] for m in methods], "o", color=BLUE, ms=6, label="solution error")
    axes[2].semilogy(x2, [full.loc[m, "relative_residual"] for m in methods], "D", color=RED, ms=6, label="PDE residual")
    axes[2].set_title("(c) Full cached solve")
    axes[2].set_ylim(4e-4, 3e-2)
    axes[2].set_xlim(-0.4, 1.4)
    axes[2].set_xticks(x2, ["Rounded", "Greedy"])
    axes[2].legend(frameon=False, loc="upper left")

    for ax in axes[:2]:
        ax.set_xlim(-0.45, len(x) - 0.55)
        ax.set_xticks(x, d["short"].tolist(), rotation=27, ha="right")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.18)
    finish(fig, "figure_05_b8_optimizer.png")


def figure_06() -> None:
    """Offline audit of the mapped Dirac-3 inverse-column instances."""
    summary = pd.read_csv(DATA / "dirac_plot_summary.csv")
    scatter = pd.read_csv(DATA / "dirac_scatter.csv")
    labels = summary["short"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))

    x = np.arange(len(summary))
    axes[0].bar(x, summary["exact_match_fraction"], width=0.55, color=BLUE, alpha=0.82)
    axes[0].set_title("(a) Exact recovery")
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlim(-0.45, len(x) - 0.55)
    axes[0].set_xticks(x, labels, rotation=24, ha="right")

    markers = {1: "o", 2: "s", 3: "^"}
    colors = {1: BLUE, 2: TEAL, 3: ORANGE}
    for code in (1, 2, 3):
        sub = scatter[scatter["pde_code"] == code]
        axes[1].semilogy(sub["idx"], sub["residual_ratio"], linestyle="none", marker=markers[code], color=colors[code], ms=4)
    axes[1].axhline(1.0, color=GRAY, linestyle="--", linewidth=1.0)
    axes[1].set_title("(b) Residual ratio")
    axes[1].set_xlabel("Mapped instance")
    # Deliberately omit the eta equation from the y-axis label; its definition belongs
    # in the accompanying documentation and data dictionary, not inside the compact plot.
    axes[1].set_ylim(0.8, 10)
    axes[1].set_xlim(0, 61)

    for code in (1, 2, 3):
        sub = scatter[scatter["pde_code"] == code]
        gap = np.maximum(sub["normalized_objective_gap"].to_numpy(), 1e-9)
        axes[2].loglog(gap, sub["local_inverse_residual_dirac"], linestyle="none", marker=markers[code], color=colors[code], ms=4)
    axes[2].set_title("(c) Gap vs. residual")
    axes[2].set_xlabel("Normalized objective gap")
    axes[2].set_ylabel(r"$r_S^{\rm D}$")
    axes[2].set_xlim(5e-10, 5e-5)
    axes[2].set_ylim(7e-5, 8e-3)

    for ax in axes:
        ax.grid(True, which="both", alpha=0.18)
    finish(fig, "figure_06_dirac3_audit.png")


def figure_07() -> None:
    """End-to-end PDE reconstructions from archived Dirac-3 samples."""
    heat = pd.read_csv(DATA / "dirac_heat.csv")
    pois = pd.read_csv(DATA / "dirac_poisson_slices.csv")
    kg = pd.read_csv(DATA / "dirac_kg.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.1))

    ax = axes[0, 0]
    ax.plot(heat["index"], heat["dense_ground_truth"], color=BLUE, label="dense")
    ax.plot(heat["index"], heat["qci_quantum"], "o-", color=ORANGE, ms=3, label="Dirac-3 factors")
    ax.set_title("(a) Heat 1D")
    ax.set_xlabel("Grid index")
    ax.set_ylabel(r"$u$")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    s1 = pois[pois["slice"] == 1]
    s2 = pois[pois["slice"] == 2]
    ax.plot(s1["x_index"], s1["dense"], color=BLUE)
    ax.plot(s1["x_index"], s1["hardware"], "o-", color=ORANGE, ms=3)
    ax.plot(s2["x_index"], s2["dense"], "--", color=TEAL)
    ax.plot(s2["x_index"], s2["hardware"], "s--", color=RED, ms=3)
    ax.set_title("(b) Poisson 2D slices")
    ax.set_xlabel(r"$x$-index")
    ax.set_ylabel(r"$u$")

    ax = axes[1, 0]
    ax.plot(kg["space_index"], kg["u_dense_ground_truth"], color=BLUE)
    ax.plot(kg["space_index"], kg["u_qci_quantum"], "o-", color=ORANGE, ms=3)
    ax.set_title(r"(c) Klein--Gordon: $u$")
    ax.set_xlabel("Spatial index")
    ax.set_ylabel(r"$u$")

    ax = axes[1, 1]
    ax.plot(kg["space_index"], kg["v_dense_ground_truth"], color=BLUE)
    ax.plot(kg["space_index"], kg["v_qci_quantum"], "o-", color=ORANGE, ms=3)
    ax.set_title(r"(d) Klein--Gordon: $v$")
    ax.set_xlabel("Spatial index")
    ax.set_ylabel(r"$v$")

    for ax in axes.flat:
        ax.margins(x=0.04, y=0.06)
        ax.grid(True, alpha=0.18)
    finish(fig, "figure_07_dirac3_reconstructions.png")


def main() -> None:
    configure_matplotlib()
    figure_02()
    figure_03()
    figure_04()
    figure_05()
    figure_06()
    figure_07()
    print(f"Generated result figures in: {OUT}")


if __name__ == "__main__":
    main()
