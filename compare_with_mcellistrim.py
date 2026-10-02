#!/usr/bin/env python3
"""
Compare the density of states of the tight-binding model and of the paper's
k.p model with the DOS curves published by McEllistrim, Garcia-Ruiz, Goodwin
and Fal'ko, arXiv:2302.07374 (2023), Fig. 1 (ABAB, ABCA, ABCB) and, when the
digitised data are available, Fig. 7 (trilayers and pentalayers).

Usage:
    python compare_with_mcellistrim.py [--dk 0.001] [--out dos_comparison_mcellistrim2023.png]
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import multilayer_graphene as mlg

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "data")
COLORS = {"tb": "#2a78d6", "kp": "#eb6834", "paper": "#1baf7a"}
LABELS = {"tb": "tight binding (this work)", "kp": "k·p, Eq. (1) of the paper", "paper": "paper, digitised"}
REFERENCE_ENERGIES = (-0.7, -0.5, -0.3, 0.3, 0.5, 0.7)


def read_digitised(path):
    """Read a digitised-curve CSV: '#' comment lines, one header line, four columns."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("energy"):
                continue
            rows.append([float(v) for v in line.split(",")])
    return np.array(rows)


def load_paper(path):
    data = read_digitised(path)
    return data[:, 0], data[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dk", type=float, default=1.0e-3, help="k mesh spacing (1/A)")
    ap.add_argument("--bins", type=int, default=800, help="energy bins")
    ap.add_argument("--out", default="dos_comparison_mcellistrim2023.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    mlg.use_paper_parameters()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "mcellistrim2023_fig*_dos.csv")))
    panels = []
    for f in files:
        m = re.search(r"fig(\d)_([ABC]+)_dos", os.path.basename(f))
        panels.append((int(m.group(1)), m.group(2), f))
    panels.sort(key=lambda t: (t[0], len(t[1]), t[1]))

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 5.2 * nrow), squeeze=False)
    print(f"{'stack':7s} {'model':4s} " + " ".join(f"{'E=%+.1f' % e:>9s}" for e in REFERENCE_ENERGIES) + "   (relative deviation from the paper)")
    for i, (fignum, seq, path) in enumerate(panels):
        ax = axes[i // ncol, i % ncol]
        E_paper, dos_paper = load_paper(path)
        e_lim = max(abs(E_paper.min()), abs(E_paper.max())) + 0.03
        ax.plot(dos_paper / 1e14, E_paper, color=COLORS["paper"], linewidth=2.2, label=LABELS["paper"], alpha=0.9)
        for model in ("kp", "tb"):
            d = mlg.calculate_density_of_states((-e_lim, e_lim), stacking_type=seq, model=model,
                                                dk=args.dk, num_energy_points=args.bins)
            ax.plot(d["dos_per_cm2"] / 1e14, d["energy"], color=COLORS[model], linewidth=1.0, label=LABELS[model])
            devs = []
            for e in REFERENCE_ENERGIES:
                if abs(e) > e_lim - 0.05:
                    devs.append(np.nan)
                    continue
                mine = np.interp(e, d["energy"], d["dos_per_cm2"])
                theirs = np.interp(e, E_paper, dos_paper)
                devs.append(100.0 * (mine / theirs - 1.0))
            print(f"{seq:7s} {model:4s} " + " ".join(f"{v:+8.1f}%" if np.isfinite(v) else "      n/a" for v in devs))
        edge = max(np.interp(-e_lim + 0.05, d["energy"], d["dos_per_cm2"]),
                   np.interp(e_lim - 0.05, d["energy"], d["dos_per_cm2"]))
        ax.set_xlim(0.0, 1.08 * max(np.nanmax(dos_paper), edge) / 1e14)
        ax.set_ylim(-e_lim, e_lim)
        title = f"{seq}  (paper Fig. {fignum})"
        if fignum == 7 and seq == "ABA":
            title += "\npaper's DOS axis is 2x too small"
        ax.set_title(title)
        ax.set_xlabel("DOS (10¹⁴ eV⁻¹ cm⁻²)")
        if i % ncol == 0:
            ax.set_ylabel("Energy (eV)")
        if i == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=False)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    fig.suptitle("Density of states: SWMcC model with the parameters of McEllistrim et al. (2023)", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(args.out, dpi=200)
    print(f"figure written to {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
