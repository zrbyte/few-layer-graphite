#!/usr/bin/env python3
"""
Electronic Raman scattering (ERS) background of bulk graphite: density of
states of hexagonal (Bernal, AB) and rhombohedral (ABC) graphite.

In graphitic systems the ERS continuum is made of interband electron-hole
excitations with negligible momentum transfer and a nearly constant matrix
element, which connect the mirror bands n- -> n+ (Kashuba and Fal'ko, PRB 80,
241404 (2009); Ponosov et al., PRB 91, 195435 (2015), Eq. (1); Garcia-Ruiz et
al., Nano Lett. 19, 6152 (2019), Eq. (5)).  The spectral density at Raman
shift w therefore follows the density of states at E_F -+ hbar w / 2, so that a
spectrometer covering w_min ... w_max probes |E - E_F| between hbar w_min / 2
and hbar w_max / 2.

The bulk curves are followed by the DOS of finite slabs (20, 50 and 100
layers by default) of both stackings, which show how the subbands and the
surface band of a film approach the bulk.

A last set of plots shows the bulk and the slab curves with the energy scale
broadened at 300 K, that is convolved with the derivative of the Fermi function.

Writes four figures, two CSV tables and an HTML report (LaTeX set by MathJax):
    <prefix>_comparison.png     DOS of both polytypes on one graph
    <prefix>_raman_window.png   the same, with the accessible energies shaded
    <prefix>_slabs.png          DOS of the finite slabs over the bulk curves
    <prefix>_thermal.png        bulk and slabs, thermally broadened
    <prefix>.csv                E - E_F and the two bulk DOS curves
    <prefix>_slabs.csv          E - E_F and the DOS of every slab
    <prefix>_report.html        figures, numbers and method on one page

Usage:
    python graphite_ers_background.py [--raman-min 80] [--raman-max 6200]
                                      [--slabs 20 50 100] [--temperature 300]
                                      [--prefix examples/graphite_ers_dos]
                                      [--from-csv examples/graphite_ers_dos.csv]
"""
import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import multilayer_graphene as mlg
import bulk_graphite as bg

EV_PER_INVERSE_CM = 1.239841984e-4      # h c
K_BOLTZMANN = 8.617333262e-5            # eV / K
STACKS = (("ab", "hexagonal (AB)", "#2a78d6"), ("abc", "rhombohedral (ABC)", "#eb6834"))
# the 2 x 2 rhombohedral problem is cheap and its Dirac points need the finer mesh
REFINE_FACTOR = {"ab": 1, "abc": 4}
SLAB_PATTERN = {"ab": "aba", "abc": "abc"}      # periodic patterns of multilayer_graphene
BLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
INK, INK_SOFT, GRID, SURFACE, WINDOW = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#eda100"


def compute(e_max=1.0, n_kz=240, dk=2.0e-3, refine=8, bin_width=1.0e-3, sigma=0.0):
    """DOS of both polytypes on a common grid of E - E_F (eV)."""
    pad = 0.1                           # room for the Fermi levels of the two polytypes
    energy = np.arange(-e_max, e_max + 0.5 * bin_width, bin_width)
    table = {"energy": energy}
    info = {"parameters": mlg.get_parameters(), "d_layer": bg.D_LAYER, "n_kz": n_kz, "dk": dk,
            "bin_width": bin_width, "gaussian_sigma": sigma,
            "atoms_per_cm3": bg.atomic_density() * 1.0e24}
    for key, label, _ in STACKS:
        d = bg.calculate_bulk_density_of_states(
            (-e_max - pad, e_max + pad), key, n_kz=n_kz, dk=dk, refine=refine * REFINE_FACTOR[key],
            num_energy_points=int(round(2.0 * (e_max + pad) / bin_width)),
            gaussian_sigma=sigma, verbose=True)
        table[key] = np.interp(energy, d["energy"] - d["fermi_energy"], d["dos_per_atom"])
        info[key] = {"label": label, "fermi_energy_eV": d["fermi_energy"],
                     "carrier_density_cm3": d["carrier_density_cm3"],
                     "refine": refine * REFINE_FACTOR[key]}
    return table, info


def save_csv(path, table, info):
    with open(path, "w") as fh:
        fh.write(f"# metadata: {json.dumps(info, sort_keys=True)}\n")
        fh.write("energy_minus_EF_eV,dos_hexagonal_states_per_eV_atom,dos_rhombohedral_states_per_eV_atom\n")
        for row in zip(table["energy"], table["ab"], table["abc"]):
            fh.write(",".join(f"{v:.8g}" for v in row) + "\n")


def load_csv(path):
    with open(path) as fh:
        info = json.loads(fh.readline().split("# metadata:", 1)[1])
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    return {"energy": data[:, 0], "ab": data[:, 1], "abc": data[:, 2]}, info


def _slab_dos(job):
    """DOS per atom of one slab on a grid of E - E_F and its Fermi level; runs in a worker process."""
    n_layers, key, energy, dk, k_max, params = job
    mlg.set_parameters(**params)
    bin_width = energy[1] - energy[0]
    e_lim = energy[-1] + 0.05           # room for the Fermi level of the slab
    d = mlg.calculate_density_of_states((-e_lim, e_lim), n_layers, SLAB_PATTERN[key], dk=dk, k_max=k_max,
                                        num_energy_points=int(round(2.0 * e_lim / bin_width)))
    # charge neutrality: n_layers of the 2 n_layers bands are filled inside the k window
    net = d["cumulative_per_area"] - n_layers * (2.0 * k_max) ** 2 / np.pi ** 2
    fermi = float(np.interp(0.0, net, d["energy_edges"]))
    return np.interp(energy, d["energy"] - fermi, d["dos_per_unit_cell"] / (2.0 * n_layers)), fermi


def compute_slabs(layers, e_max=0.5, dk=1.0e-3, k_max=0.18, bin_width=1.0e-3):
    """
    DOS per atom of finite slabs of both stackings on a common grid of
    E - E_F (eV), from ``multilayer_graphene.calculate_density_of_states``.
    The slabs are diagonalised in parallel worker processes.
    """
    layers = sorted(int(n) for n in layers)
    energy = np.arange(-e_max, e_max + 0.5 * bin_width, bin_width)
    jobs = [(n, key, energy, dk, k_max, mlg.get_parameters()) for n in reversed(layers) for key, _, _ in STACKS]
    # One BLAS thread per worker, set before the workers import NumPy: threaded
    # LAPACK calls from several processes at once slow each other down a
    # hundredfold on matrices of this size.
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(BLAS_THREAD_VARIABLES, "1"))
    try:
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_slab_dos, jobs))
    finally:
        for name, value in saved.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
    slabs = {"energy": energy}
    info = {"layers": layers, "dk": dk, "k_max": k_max, "bin_width": bin_width,
            "parameters": mlg.get_parameters(), "fermi_energy_eV": {}}
    for (n, key, *_), (dos, fermi) in sorted(zip(jobs, results), key=lambda item: item[0][:2]):
        slabs[f"{key}_{n}"] = dos
        info["fermi_energy_eV"][f"{key}_{n}"] = fermi
        print(f"slab DOS: {n:4d} layers {SLAB_PATTERN[key]}, E_F = {1e3 * fermi:+.2f} meV")
    return slabs, info


def save_slabs_csv(path, slabs, info):
    columns = list(slabs)
    with open(path, "w") as fh:
        fh.write(f"# metadata: {json.dumps({**info, 'columns': columns}, sort_keys=True)}\n")
        fh.write(",".join(["energy_minus_EF_eV"] + [f"dos_{c}_layers_states_per_eV_atom" for c in columns[1:]]) + "\n")
        for row in zip(*(slabs[c] for c in columns)):
            fh.write(",".join(f"{v:.8g}" for v in row) + "\n")


def load_slabs_csv(path):
    with open(path) as fh:
        info = json.loads(fh.readline().split("# metadata:", 1)[1])
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    return {c: data[:, i] for i, c in enumerate(info.pop("columns"))}, info


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def _draw_dos(ax, table, info, xlim, linewidth=1.8):
    sel = (table["energy"] >= xlim[0]) & (table["energy"] <= xlim[1])
    for key, label, color in STACKS:
        ax.plot(table["energy"][sel], 100.0 * table[key][sel], color=color, linewidth=linewidth, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 1.06 * 100.0 * max(table[key][sel].max() for key, _, _ in STACKS))
    ax.set_xlabel("E − E$_F$ (eV)", color=INK)
    ax.set_ylabel("DOS (10$^{-2}$ states eV$^{-1}$ atom$^{-1}$)", color=INK)
    per_cm3 = info["atoms_per_cm3"] * 1.0e-2 / 1.0e21
    right = ax.secondary_yaxis("right", functions=(lambda y: y * per_cm3, lambda y: y / per_cm3))
    right.set_ylabel("DOS (10$^{21}$ eV$^{-1}$ cm$^{-3}$)", color=INK)
    right.tick_params(colors=INK_SOFT, labelsize=9)
    right.spines["right"].set_color(INK_SOFT)
    _style(ax)


def plot_comparison(table, info, path, e_max=1.0, inset=0.1):
    """Figure 1: the two DOS curves, with a close-up of the Fermi level."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    _draw_dos(ax, table, info, (-e_max, e_max))
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, labelcolor=INK)
    ax.set_title("Bulk graphite: density of states of the SWMcC model", color=INK, fontsize=11, loc="left")

    box = ax.inset_axes([0.36, 0.50, 0.30, 0.32])
    sel = np.abs(table["energy"]) <= inset
    for key, _, color in STACKS:
        box.plot(1.0e3 * table["energy"][sel], 100.0 * table[key][sel], color=color, linewidth=1.5)
    box.set_xlim(-1.0e3 * inset, 1.0e3 * inset)
    box.set_ylim(0.0, 1.1 * 100.0 * max(table[key][sel].max() for key, _, _ in STACKS))
    box.set_xlabel("E − E$_F$ (meV)", color=INK_SOFT, fontsize=8, labelpad=1)
    _style(box)
    box.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def plot_raman_window(table, info, path, raman_min, raman_max, e_max=0.5):
    """Figure 2: the DOS with the energies reached by the spectrometer shaded."""
    e_lo = 0.5 * raman_min * EV_PER_INVERSE_CM
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    fig, ax = plt.subplots(figsize=(7.2, 4.9), facecolor=SURFACE)
    _draw_dos(ax, table, info, (-e_max, e_max))
    for sign in (-1.0, 1.0):
        ax.axvspan(sign * e_lo, sign * e_hi, color=WINDOW, alpha=0.16, linewidth=0)
        ax.axvline(sign * e_hi, color=INK_SOFT, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.plot([], [], color=WINDOW, alpha=0.35, linewidth=8, solid_capstyle="butt",
            label=f"Raman shift {raman_min:g}–{raman_max:g} cm$^{{-1}}$:  "
                  f"|E − E$_F$| = {1e3 * e_lo:.0f}–{1e3 * e_hi:.0f} meV")
    ax.legend(loc="upper center", frameon=False, fontsize=9, labelcolor=INK)

    # Raman shift of the n- -> n+ excitation that connects -E and +E
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    shifts = np.arange(0.0, 2.0 * e_max / EV_PER_INVERSE_CM + 1.0, 2000.0)
    ticks = np.concatenate([-0.5 * shifts[:0:-1], 0.5 * shifts]) * EV_PER_INVERSE_CM
    top.set_xticks(ticks)
    top.set_xticklabels([f"{abs(2.0 * t) / EV_PER_INVERSE_CM:.0f}" for t in ticks])
    top.set_xlabel("Raman shift ħω = 2|E − E$_F$| (cm$^{-1}$)", color=INK)
    top.tick_params(colors=INK_SOFT, labelsize=9)
    for side in ("top", "right", "left", "bottom"):
        top.spines[side].set_visible(side == "top")
    top.spines["top"].set_color(INK_SOFT)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def _raman_axis(ax, e_max):
    """Top axis with the Raman shift of the n- -> n+ excitation that connects -E and +E."""
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    shifts = np.arange(0.0, 2.0 * e_max / EV_PER_INVERSE_CM + 1.0, 2000.0)
    ticks = np.concatenate([-0.5 * shifts[:0:-1], 0.5 * shifts]) * EV_PER_INVERSE_CM
    top.set_xticks(ticks)
    top.set_xticklabels([f"{abs(2.0 * t) / EV_PER_INVERSE_CM:.0f}" for t in ticks])
    top.set_xlabel("Raman shift ħω = 2|E − E$_F$| (cm$^{-1}$)", color=INK)
    top.tick_params(colors=INK_SOFT, labelsize=9)
    for side in ("top", "right", "left", "bottom"):
        top.spines[side].set_visible(side == "top")
    top.spines["top"].set_color(INK_SOFT)
    return top


def bulk_on_slab_axis(slabs, slab_info, table, info, key, n):
    """Bulk DOS at the same absolute energy as the slab curve, on the slab's axis of E - E_F."""
    shift = slab_info["fermi_energy_eV"][f"{key}_{n}"] - info[key]["fermi_energy_eV"]
    return np.interp(slabs["energy"] + shift, table["energy"], table[key])


def plot_slabs(slabs, slab_info, table, info, path, raman_min, raman_max, e_max=0.5, zoom=0.1):
    """
    Figure 3: DOS of the finite slabs, one row per thickness.  Left: the range
    of Figure 2 with the bulk curves for reference; right: close-up of the
    Fermi level.  Energies are measured from the Fermi level of each slab and
    the bulk curves are drawn at the same absolute energy.
    """
    layers = slab_info["layers"]
    e_lo = 0.5 * raman_min * EV_PER_INVERSE_CM
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    energy = slabs["energy"]
    wide, near = np.abs(energy) <= e_max, np.abs(energy) <= zoom
    keys = [key for key, _, _ in STACKS]
    y_wide = 1.06 * 100.0 * max(table[key][np.abs(table["energy"]) <= e_max].max() for key in keys)
    y_near = 1.08 * 100.0 * max(slabs[f"{key}_{n}"][near].max() for key in keys for n in layers)

    fig, axes = plt.subplots(len(layers), 2, figsize=(10.4, 0.6 + 3.0 * len(layers)), facecolor=SURFACE,
                             squeeze=False, gridspec_kw={"width_ratios": [1.65, 1.0]})
    for row, n in enumerate(layers):
        left, right = axes[row]
        bulk = {key: bulk_on_slab_axis(slabs, slab_info, table, info, key, n) for key in keys}
        for ax, sel, scale in ((left, wide, 1.0), (right, near, 1.0e3)):
            for key, label, color in STACKS:
                ax.plot(scale * energy[sel], 100.0 * slabs[f"{key}_{n}"][sel], color=color, linewidth=1.2,
                        label=f"{label}, {n} layers" if ax is left else None)
                ax.plot(scale * energy[sel], 100.0 * bulk[key][sel], color=INK, linewidth=0.8,
                        linestyle=(0, (4, 2)), label="bulk" if ax is left and key == "abc" else None)
            for sign in (-1.0, 1.0):
                ax.axvspan(scale * sign * e_lo, scale * sign * min(e_hi, 2.0 * e_max), color=WINDOW, alpha=0.16, linewidth=0)
            _style(ax)
        left.set_xlim(-e_max, e_max)
        left.set_ylim(0.0, y_wide)
        right.set_xlim(-1.0e3 * zoom, 1.0e3 * zoom)
        right.set_ylim(0.0, y_near)
        left.legend(loc="upper center", frameon=False, fontsize=8.5, labelcolor=INK, ncol=1)
        left.set_ylabel("DOS (10$^{-2}$ states eV$^{-1}$ atom$^{-1}$)", color=INK)
        if row == len(layers) - 1:
            left.set_xlabel("E − E$_F$ (eV)", color=INK)
            right.set_xlabel("E − E$_F$ (meV)", color=INK)
    _raman_axis(axes[0, 0], e_max)
    axes[0, 1].set_title("close-up of the Fermi level", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def thermal_broadening(energy, dos, temperature):
    """
    Convolve a DOS on a uniform energy grid (eV) with the derivative of the
    Fermi function at the given temperature (K),

        -df/dE = sech^2(E / 2kT) / 4kT        (FWHM 3.53 kT).

    The kernel is integrated over each energy bin and the curve is continued
    by its end values, so a constant DOS stays constant; values closer than
    about 5 kT to the ends of the grid depend on that continuation.
    """
    kT = K_BOLTZMANN * temperature
    h = energy[1] - energy[0]
    m = int(np.ceil(12.0 * kT / h))
    edges = (np.arange(-m, m + 2) - 0.5) * h
    kernel = -np.diff(1.0 / (np.exp(edges / kT) + 1.0))
    padded = np.pad(dos, m + 1, mode="edge")
    return np.convolve(padded, kernel / kernel.sum(), mode="same")[m + 1:-(m + 1)]


def thermal_curves(table, info, slabs, slab_info, temperature):
    """
    Thermally broadened DOS per atom on the energy grid of the bulk table:
    {'bulk': {key: D}, n_layers: {key: D}, ...}, each against E - E_F of its
    own system.  Beyond its own energy range a slab curve is continued by the
    bulk curve at the same absolute energy before it is broadened.
    """
    energy = table["energy"]
    keys = [key for key, _, _ in STACKS]
    curves = {"bulk": {key: thermal_broadening(energy, table[key], temperature) for key in keys}}
    for n in (slab_info["layers"] if slabs is not None else []):
        inside = (energy >= slabs["energy"][0]) & (energy <= slabs["energy"][-1])
        curves[n] = {}
        for key in keys:
            shift = slab_info["fermi_energy_eV"][f"{key}_{n}"] - info[key]["fermi_energy_eV"]
            continued = np.where(inside, np.interp(energy, slabs["energy"], slabs[f"{key}_{n}"]),
                                 np.interp(energy + shift, energy, table[key]))
            curves[n][key] = thermal_broadening(energy, continued, temperature)
    return curves


def bulk_on_broadened_slab_axis(curves, table, info, slab_info, key, n):
    """Broadened bulk DOS at the same absolute energy as the broadened slab curve."""
    shift = slab_info["fermi_energy_eV"][f"{key}_{n}"] - info[key]["fermi_energy_eV"]
    return np.interp(table["energy"] + shift, table["energy"], curves["bulk"][key])


def plot_thermal(curves, table, info, slab_info, path, raman_min, raman_max, temperature, e_max=0.45):
    """
    Figure 4: the DOS broadened by the thermal kernel, one panel for the bulk
    and one per slab.  Dashed in the slab panels: the broadened bulk curves at
    the same absolute energy.
    """
    energy = table["energy"]
    sel = np.abs(energy) <= e_max
    e_lo = 0.5 * raman_min * EV_PER_INVERSE_CM
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    panels = list(curves)
    ncol = min(2, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 0.7 + 3.4 * nrow), facecolor=SURFACE, squeeze=False)
    y_max = 1.06 * 100.0 * max(curves[p][key][sel].max() for p in panels for key, _, _ in STACKS)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(panels):
            ax.axis("off")
            continue
        panel = panels[i]
        for key, label, color in STACKS:
            ax.plot(energy[sel], 100.0 * curves[panel][key][sel], color=color, linewidth=1.8, label=label)
            if panel != "bulk":
                bulk = bulk_on_broadened_slab_axis(curves, table, info, slab_info, key, panel)
                ax.plot(energy[sel], 100.0 * bulk[sel], color=INK, linewidth=0.8, linestyle=(0, (4, 2)),
                        label="bulk" if key == "abc" else None)
        for sign in (-1.0, 1.0):
            ax.axvspan(sign * e_lo, sign * min(e_hi, e_max), color=WINDOW, alpha=0.16, linewidth=0)
        ax.set_xlim(-e_max, e_max)
        ax.set_ylim(0.0, y_max)
        _style(ax)
        ax.legend(loc="upper center", frameon=False, fontsize=8.5, labelcolor=INK,
                  title="bulk" if panel == "bulk" else f"{panel} layers", title_fontproperties={"weight": "bold", "size": 9})
        if i // ncol == 0:
            _raman_axis(ax, e_max)
        if i + ncol >= len(panels):
            ax.set_xlabel("E − E$_F$ (eV)", color=INK)
        if i % ncol == 0:
            ax.set_ylabel(f"DOS at {temperature:g} K (10$^{{-2}}$ states eV$^{{-1}}$ atom$^{{-1}}$)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def window_mean(energy, dos, raman_min, raman_max):
    """Mean DOS over the energies E_F -+ hbar w / 2 reached by the spectrometer."""
    sel = (np.abs(energy) >= 0.5 * raman_min * EV_PER_INVERSE_CM) & (np.abs(energy) <= 0.5 * raman_max * EV_PER_INVERSE_CM)
    return float(dos[sel].mean())


def dos_at_shift(table, shift):
    """DOS (states/(eV atom)) of both polytypes at E_F -+ hbar w / 2, averaged over the two signs."""
    e = 0.5 * shift * EV_PER_INVERSE_CM
    return tuple(0.5 * (np.interp(e, table["energy"], table[key]) + np.interp(-e, table["energy"], table[key]))
                 for key, _, _ in STACKS)


def shift_rows(table, raman_min, raman_max):
    """(Raman shift, |E - E_F| in meV, DOS hexagonal, DOS rhombohedral, ratio) across the window."""
    shifts = sorted({raman_min, raman_max} | {v for v in (200.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
                                               if raman_min < v < raman_max})
    rows = []
    for shift in shifts:
        hexa, rhombo = dos_at_shift(table, shift)
        rows.append((shift, 0.5e3 * shift * EV_PER_INVERSE_CM, hexa, rhombo, rhombo / hexa))
    return rows


def crossing_shift(table):
    """Raman shift (1/cm) above which the rhombohedral DOS exceeds the hexagonal one."""
    shifts = np.arange(100.0, 2.0 * table["energy"].max() / EV_PER_INVERSE_CM, 10.0)
    diff = np.array([np.subtract(*dos_at_shift(table, w)) for w in shifts])
    change = np.nonzero((diff[:-1] > 0.0) & (diff[1:] <= 0.0))[0]
    return float(shifts[change[0] + 1]) if change.size else np.nan


def report(table, info, raman_min, raman_max):
    for key, label, _ in STACKS:
        print(f"{label:19s} E_F = {1e3 * info[key]['fermi_energy_eV']:+7.2f} meV, "
              f"n = p = {info[key]['carrier_density_cm3']:.2e} cm^-3, "
              f"DOS(E_F) = {np.interp(0.0, table['energy'], table[key]):.2e} states/(eV atom)")
    print("DOS at E_F -+ hbar w / 2, averaged over the two signs (10^-2 states/(eV atom)):")
    print(f"{'Raman shift (1/cm)':>19s} {'|E-E_F| (meV)':>14s} {'hexagonal':>10s} {'rhombohedral':>13s} {'ratio':>6s}")
    for shift, e, hexa, rhombo, ratio in shift_rows(table, raman_min, raman_max):
        print(f"{shift:19.0f} {e:14.1f} {100 * hexa:10.3f} {100 * rhombo:13.3f} {ratio:6.2f}")
    print(f"the two curves cross at a Raman shift of {crossing_shift(table):.0f} 1/cm")


def _listing(values):
    """'a, b and c' for a list of strings."""
    return ", ".join(values[:-1]) + " and " + values[-1] if len(values) > 1 else values[0]


def ers_peak(n, n_layers, gamma1):
    """n- -> n+ excitation energy (eV) of an n_layers rhombohedral film, Garcia-Ruiz et al. (2019), Eq. (1)."""
    return 4.0 * gamma1 * np.sin((n + 0.5) * np.pi / (2 * n_layers + 1))


def subband_peak_pair(energy, dos, centre, expected, half_width):
    """
    Separation (eV) of the DOS peaks found within half_width of
    centre -+ expected / 2: the -n and +n subband peaks of a rhombohedral slab.
    """
    found = []
    for target in (centre - 0.5 * expected, centre + 0.5 * expected):
        sel = np.abs(energy - target) <= half_width
        found.append(energy[sel][np.argmax(dos[sel])])
    return found[1] - found[0]


# LaTeX fragments of the report; the page loads MathJax to typeset them (SVG
# output, which is one script and fetches no font files at run time)
MATHJAX = ('<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>'
           '<style>mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; }</style>')
CM1 = r"\(\mathrm{cm^{-1}}\)"
CM3 = r"\(\mathrm{cm^{-3}}\)"
PER_ANGSTROM = r"\(\text{Å}^{-1}\)"
PER_EV_ATOM = r"\(\mathrm{eV^{-1}\,atom^{-1}}\)"
EF = r"\(E_F\)"
DE = r"\(|E - E_F|\)"
PAIR = r"\(n^- \to n^+\)"


def _sci(x, digits=1):
    """A number as LaTeX in scientific notation."""
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"


def _signed(text):
    """Typographic minus for a number that is not set as math."""
    return text.replace("-", "−")


def _slab_remarks(slabs, slab_info, table, info, raman_min, raman_max):
    layers = slab_info["layers"]
    par = info["parameters"]
    energy = slabs["energy"]
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    hv = 1.5 * (par["a"] / np.sqrt(3.0)) * par["gamma0"]
    listing = _listing
    names = listing([str(n) for n in layers])

    sel = (np.abs(energy) >= 0.03) & (np.abs(energy) <= e_hi)
    deviation = [np.abs(slabs[f"ab_{n}"][sel] / bulk_on_slab_axis(slabs, slab_info, table, info, "ab", n)[sel] - 1.0).max()
                 for n in layers]
    envelope = [slabs[f"abc_{n}"][sel].mean() / bulk_on_slab_axis(slabs, slab_info, table, info, "abc", n)[sel].mean()
                for n in layers]
    first = [ers_peak(1, n, par["gamma1"]) / EV_PER_INVERSE_CM for n in layers]
    spacing = [4.0 * np.pi * par["gamma1"] / (2 * n + 1) / EV_PER_INVERSE_CM for n in layers]
    offset = [info["abc"]["fermi_energy_eV"] - slab_info["fermi_energy_eV"][f"abc_{n}"] for n in layers]
    surface_states = 2.0 * (par["gamma1"] / hv) ** 2 * mlg.unit_cell_area(par["a"]) / np.pi

    thin = layers[0]
    merged = [str(n) for n, w in zip(layers, spacing) if w < 500.0]
    merging = (f"The ladders of the {_listing([n + '-' for n in merged[:-1]] + merged[-1:])}-layer slabs "
               "are finer than that and merge" if merged
               else "A ladder finer than that merges")
    check = ""
    if 2 * thin + 1 < 80:       # subbands far enough apart to find the peaks by their expected position
        pairs = [(subband_peak_pair(energy, slabs[f"abc_{thin}"], offset[0], ers_peak(n, thin, par["gamma1"]),
                                    np.pi * par["gamma1"] / (2 * thin + 1) / 2.0), ers_peak(n, thin, par["gamma1"]))
                 for n in (1, 2)]
        check = (f" In the {thin}-layer DOS the first two pairs of peaks are "
                 f"{pairs[0][0] * 1e3:.0f} and {pairs[1][0] * 1e3:.0f} meV apart; the formula gives "
                 f"{pairs[0][1] * 1e3:.0f} and {pairs[1][1] * 1e3:.0f} meV.")

    return (
        "<ul>"
        "<li><b>Hexagonal slabs follow the bulk curve.</b> Between 30 meV and the upper edge of the window the "
        f"largest deviation is {listing([f'{100 * d:.1f} %' for d in deviation])} for {names} layers. "
        "The discrete subbands show only as small steps within about 20 meV of the Fermi level.</li>"
        "<li><b>Rhombohedral slabs resolve the bulk continuum into a ladder of subbands.</b> Each subband starts with a "
        r"step and a van Hove peak. The \(-n\) and \(+n\) peaks pair into the ERS peaks of García-Ruiz et al. (2019),"
        r"\[\hbar\omega_n \approx 4\gamma_1 \sin\!\left[\frac{\left(n + \frac{1}{2}\right)\pi}{2N + 1}\right].\]"
        f"The first one lies at {listing([f'{w:.0f}' for w in first])} {CM1} for {names} layers, and the peaks follow "
        r"each other every \(4\pi\gamma_1/(2N + 1) =\) "
        f"{listing([f'{w:.0f}' for w in spacing])} {CM1}.{check}</li>"
        "<li><b>Thick rhombohedral slabs give a bulk-like background.</b> The measured ERS peaks of thin films are about "
        f"500 {CM1} wide at room temperature (Pálinkás et al. 2024). {merging} into a "
        "continuum that follows the envelope of the DOS, and that envelope is the bulk curve: averaged between 30 meV and "
        f"the upper edge of the window the slab DOS is {listing([f'{100 * r:.1f} %' for r in envelope])} of the bulk DOS "
        f"for {names} layers.</li>"
        "<li><b>Surface band.</b> The plateau about 30 meV wide at the Fermi level of the rhombohedral slabs is the surface "
        r"band. It holds \(2(\gamma_1/\hbar v)^2 A_\mathrm{cell}/\pi =\) "
        f"{surface_states:.3f} states per unit cell of the film (two surfaces, "
        r"spin and valley included) whatever the thickness, so its height per atom falls as \(1/N\).</li>"
        "<li><b>Fermi level of the rhombohedral slabs.</b> The surface band pins the Fermi level "
        f"{listing([f'{1e3 * o:.0f}' for o in offset])} meV below the bulk Dirac energy for {names} layers. The dashed bulk "
        f"curves are drawn at the same absolute energy, which puts their minimum to the right of zero. The {PAIR} "
        f"pairs are centred near the Dirac energy and not on {EF}, so for the subbands the shaded "
        r"window should be read as shifted by that amount. At low temperature the pairs with \(\hbar\omega/2\) smaller "
        f"than the offset (shifts below about {round(2.0 * min(offset) / EV_PER_INVERSE_CM, -1):.0f} {CM1}) are Pauli "
        "blocked. The offset belongs to the non-interacting model, which leaves out the charge redistribution between "
        "surface and interior.</li>"
        "</ul>")


def _slab_section(slabs, slab_info, table, info, raman_min, raman_max, fig_slabs):
    """Report section on the finite slabs: figure, table and remarks."""
    from make_report import img_tag, table as html_table

    layers = slab_info["layers"]
    e_lo, e_hi = (0.5e3 * w * EV_PER_INVERSE_CM for w in (raman_min, raman_max))
    names = _listing([str(n) for n in layers])
    rows = []
    for n in layers:
        hexa, rhombo = (window_mean(slabs["energy"], slabs[f"{key}_{n}"], raman_min, raman_max) for key, _, _ in STACKS)
        rows.append((f"{n} layers", _signed(f"{slab_info['fermi_energy_eV'][f'ab_{n}'] * 1e3:+.1f}"),
                     _signed(f"{slab_info['fermi_energy_eV'][f'abc_{n}'] * 1e3:+.1f}"),
                     f"{100 * hexa:.3f}", f"{100 * rhombo:.3f}", f"{rhombo / hexa:.3f}"))
    hexa, rhombo = (window_mean(table["energy"], table[key], raman_min, raman_max) for key, _, _ in STACKS)
    rows.append(("bulk", _signed(f"{info['ab']['fermi_energy_eV'] * 1e3:+.1f}"),
                 _signed(f"{info['abc']['fermi_energy_eV'] * 1e3:+.1f}"),
                 f"{100 * hexa:.3f}", f"{100 * rhombo:.3f}", f"{rhombo / hexa:.3f}"))
    return [
        f"<h2>Finite slabs: {names} layers</h2>",
        r"<p>The same tight-binding model for slabs of \(N\) layers with free surfaces "
        r"(<code>multilayer_graphene.py</code>, \(2N\) bands), hexagonal (ABAB…) and rhombohedral (ABCABC…). The DOS is "
        r"divided by the \(2N\) atoms of the cell, so it can be laid over the bulk curves, and each slab has its own Fermi "
        "level from charge neutrality.</p>",
        f"<figure>{img_tag(fig_slabs, 'DOS of finite slabs of hexagonal and rhombohedral graphite')}"
        f"<figcaption>Figure 3. DOS per atom of slabs of {names} layers: hexagonal (blue) and rhombohedral (orange), "
        "energies measured from the Fermi level of each slab. Dashed: the bulk curves of Figure 1 at the same absolute "
        "energy. Left: the energy range of Figure 2, with the energies reached by "
        f"{raman_min:g}–{raman_max:g} {CM1} shaded. Right: the 100 meV around the Fermi level, on a common "
        "vertical scale.</figcaption></figure>",
        html_table(
            ["", rf"{EF} hexagonal (meV)", rf"{EF} rhombohedral (meV)", "hexagonal", "rhombohedral",
             "rhombohedral / hexagonal"],
            rows,
            f"Fermi level on the model's energy scale, and DOS averaged over {DE} = {e_lo:.0f}–{e_hi:.0f} meV "
            rf"(both signs) in \(10^{{-2}}\) states {PER_EV_ATOM}"),
        _slab_remarks(slabs, slab_info, table, info, raman_min, raman_max),
        "<p>Numerical details: in-plane mesh around K with spacing "
        f"{slab_info['dk']:g} {PER_ANGSTROM} and half-width {slab_info['k_max']:g} {PER_ANGSTROM}, triangle interpolation, "
        f"{slab_info['bin_width'] * 1e3:g} meV bins, no smoothing. "
        r"Only \(p_y \ge 0\) is diagonalised; the other half follows from \(E(p_x, -p_y) = E(p_x, p_y)\). "
        "For 20 layers, halving the mesh spacing changes single 1 meV bins at subband edges by up to "
        rf"\(0.04\times10^{{-2}}\) states {PER_EV_ATOM} and the rest of the "
        "curves by less; the thicker slabs were not repeated on a finer mesh.</p>",
    ]


def dos_ratio(table, shift):
    """Rhombohedral / hexagonal DOS ratio of a table at E_F -+ hbar w / 2."""
    hexagonal, rhombohedral = dos_at_shift(table, shift)
    return rhombohedral / hexagonal


def _thermal_section(curves, table, info, slabs, slab_info, raman_min, raman_max, temperature, fig_thermal):
    """Report section on the thermally broadened DOS of the bulk and of the slabs."""
    from make_report import img_tag, table as html_table

    energy = table["energy"]
    kT = K_BOLTZMANN * temperature
    fwhm = 2.0 * np.log(3.0 + 2.0 * np.sqrt(2.0)) * kT          # of sech^2(E / 2kT): 3.53 kT
    fwhm_shift = 2.0 * fwhm / EV_PER_INVERSE_CM                 # on the axis hbar w = 2 |E - E_F|
    panels = list(curves)
    layers = [p for p in panels if p != "bulk"]
    names = _listing([str(n) for n in layers]) if layers else ""
    tables = {p: {"energy": energy, **curves[p]} for p in panels}
    centre = int(np.argmin(np.abs(energy)))
    at_fermi = {p: {key: 100.0 * curves[p][key][centre] for key, _, _ in STACKS} for p in panels}
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    sel = np.abs(energy) <= e_hi

    shifts = sorted({raman_min, raman_max} | {v for v in (500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0)
                                             if raman_min < v < raman_max})
    rows = [[f"{w:.0f}", f"{0.5e3 * w * EV_PER_INVERSE_CM:.1f}"] + [f"{dos_ratio(tables[p], w):.2f}" for p in panels]
            for w in shifts]
    high = [w for w in shifts if w >= 2000.0]
    unchanged = max(abs(dos_ratio(tables["bulk"], w) - dos_ratio(table, w)) for w in high) if high else np.nan

    out = [
        f"<h2>Thermal broadening at {temperature:g} K</h2>",
        rf"<p>The same curves with the energy scale smeared at \(T = {temperature:g}\) K: each DOS is convolved with the "
        "derivative of the Fermi function,</p>"
        r"\[D_T(E) = \int D(E')\left(-\frac{\partial f}{\partial E}\right)_{E - E'} dE', \qquad"
        r" -\frac{\partial f}{\partial E} = \frac{1}{4k_B T}\,\operatorname{sech}^2\!\left(\frac{E}{2k_B T}\right),\]"
        rf"<p>with \(k_B T = {kT * 1e3:.1f}\) meV. The kernel has a full width at half maximum of "
        rf"\(3.53\,k_B T = {fwhm * 1e3:.0f}\) meV on the energy axis, which is "
        rf"\(7.05\,k_B T\) or {fwhm_shift:.0f} {CM1} on the Raman-shift axis "
        r"\(\hbar\omega = 2|E - E_F|\).</p>",
        f"<figure>{img_tag(fig_thermal, 'Thermally broadened DOS of bulk graphite and of finite slabs')}"
        f"<figcaption>Figure 4. DOS broadened at {temperature:g} K: hexagonal (blue) and rhombohedral (orange), for the bulk"
        + (f" and for slabs of {names} layers" if layers else "")
        + ". Energies are measured from the Fermi level of each system and the energies reached by "
        f"{raman_min:g}–{raman_max:g} {CM1} are shaded."
        + (" Dashed in the slab panels: the broadened bulk curves at the same absolute energy." if layers else "")
        + "</figcaption></figure>",
        html_table(
            [f"Raman shift ({CM1})", f"{DE} (meV)"] + ["bulk" if p == "bulk" else f"{p} layers" for p in panels],
            rows,
            f"Rhombohedral / hexagonal ratio of the DOS broadened at {temperature:g} K, at "
            r"\(E_F \mp \hbar\omega/2\) averaged over the two signs"),
    ]

    remarks = [
        "<li><b>Bulk.</b> The broadening fills the minimum of the rhombohedral DOS: at the Fermi level it is "
        f"{at_fermi['bulk']['abc']:.2f} against {at_fermi['bulk']['ab']:.2f} for hexagonal graphite "
        rf"(\(10^{{-2}}\) states {PER_EV_ATOM}). At {raman_min:g} {CM1} the ratio of the two is "
        f"{dos_ratio(tables['bulk'], raman_min):.2f}, against {dos_ratio(table, raman_min):.2f} without broadening. "
        f"From 2000 {CM1} upwards the broadening changes the ratio by less than "
        f"{max(0.01, np.ceil(100.0 * unchanged) / 100.0):.2f}, and the curves cross at "
        f"{crossing_shift(tables['bulk']):.0f} {CM1}.</li>"]
    if layers:
        deviation = [f"{100 * np.abs(curves[n]['ab'][sel] / bulk_on_broadened_slab_axis(curves, table, info, slab_info, 'ab', n)[sel] - 1.0).max():.1f} %"
                     for n in layers]
        excess = [f"{100 * (curves[n]['abc'][centre] / bulk_on_broadened_slab_axis(curves, table, info, slab_info, 'abc', n)[centre] - 1.0):.0f} %"
                  for n in layers]
        fermi_abc = [f"{at_fermi[n]['abc']:.2f}" for n in layers]
        low = [f"{dos_ratio(tables[n], raman_min):.2f}" for n in layers]
        far = max(abs(dos_ratio(tables[n], w) / dos_ratio(tables["bulk"], w) - 1.0) for n in layers for w in high)
        remarks += [
            "<li><b>The subband ladders are gone.</b> No step or van Hove peak of the slabs survives the broadening. "
            f"The hexagonal slabs stay within {_listing(deviation)} of the bulk curve for {names} layers over the whole "
            "window.</li>",
            "<li><b>What remains of the film is its surface band.</b> The rhombohedral slabs carry extra states at the "
            f"Fermi level: {_listing(fermi_abc)} for {names} layers against {at_fermi['bulk']['abc']:.2f} in the bulk, "
            f"which is {_listing(excess)} above the bulk curve at the same absolute energy. "
            r"The excess scales as \(1/N\) and sits at the low end of the window: "
            f"at {raman_min:g} {CM1} the ratio to the hexagonal DOS is {_listing(low)} against "
            f"{dos_ratio(tables['bulk'], raman_min):.2f} in the bulk, while from 2000 {CM1} upwards the slab ratios are "
            f"within {100 * far:.0f} % of the bulk ratio.</li>"]
    remarks.append(
        "<li><b>How to read this for ERS.</b> In Eq. (1) temperature enters through the occupation factors, not as a "
        "convolution of the DOS, so these curves show how much structure a thermal smearing of the energy scale leaves, "
        "not the ERS line shape. On the Raman-shift axis the kernel is "
        f"{fwhm_shift:.0f} {CM1} wide, about {fwhm_shift / 500.0:.0f} times the 500 {CM1} width of the ERS peaks "
        "that Pálinkás et al. (2024) measure at room temperature, so the figure overstates the smearing of ERS "
        "features. <code>--temperature</code> sets the kernel; "
        f"{500.0 * EV_PER_INVERSE_CM / (4.0 * np.log(3.0 + 2.0 * np.sqrt(2.0)) * K_BOLTZMANN):.0f} K gives a "
        f"width of 500 {CM1} on the Raman-shift axis.</li>")
    out.append("<ul>" + "".join(remarks) + "</ul>")
    if layers:
        tail = 1.0 / (np.exp((slabs["energy"][-1] - e_hi) / kT) + 1.0)
        out.append(
            "<p>Numerical details: the kernel is integrated over each energy bin. The slab curves end at "
            rf"\(|E - E_F| = {slabs['energy'][-1]:g}\) eV and are continued by the bulk curve at the same "
            f"absolute energy before the convolution; at the upper edge of the window {100 * tail:.1f} % of the "
            "kernel weight falls on that continuation.</p>")
    return out


def write_report(path, table, info, raman_min, raman_max, fig_comparison, fig_window,
                 slabs=None, slab_info=None, fig_slabs=None, curves=None, temperature=None, fig_thermal=None,
                 standalone=True):
    """
    Self-contained HTML page (figures embedded) in the style of make_report.py.
    The mathematics is written in LaTeX and typeset by MathJax, which the page
    loads from a CDN.  With standalone=False the document skeleton (doctype,
    html, head and body tags) is left out, for a publisher that adds its own.
    """
    from make_report import STYLE, img_tag, table as html_table

    rows = shift_rows(table, raman_min, raman_max)
    cross = crossing_shift(table)
    e_lo, e_hi = (0.5e3 * w * EV_PER_INVERSE_CM for w in (raman_min, raman_max))
    fermi = {key: (info[key]["fermi_energy_eV"], info[key]["carrier_density_cm3"],
                   float(np.interp(0.0, table["energy"], table[key]))) for key, _, _ in STACKS}
    par = info["parameters"]
    swmcc_delta = 2.0 * par["Delta_prime"] + par["gamma2"] - par["gamma5"]
    overlap = 4.0 * par["gamma4"] / par["gamma0"] * (par["gamma1"] * par["gamma3"] / par["gamma0"] + 0.5 * par["gamma2"])
    dirac = par["Delta_prime"] + 2.0 * par["gamma1"] * par["gamma4"] / par["gamma0"]

    s = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'] if standalone else []
    s += ["<title>Graphite ERS background</title>",
          '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600'
          '&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono&display=swap">',
          STYLE, "<style>.words td:last-child, .words th:last-child { text-align: left; white-space: normal; }</style>",
          MATHJAX]
    s += ["</head><body>"] if standalone else []
    s += ['<div class="wrap">',
          '<p class="eyebrow">Bulk graphite · SWMcC model · density of states</p>',
          "<h1>ERS background of bulk graphite</h1>",
          '<p class="lede">Density of states of pure hexagonal (AB) and rhombohedral (ABC) bulk graphite, as an '
          "estimate of the electronic Raman scattering (ERS) continuum of the two polytypes within the "
          f"{raman_min:g}–{raman_max:g} {CM1} range of the spectrometer.</p>",
          '<div class="key">',
          f"<div><b>{rows[0][4]:.2f} → {rows[-1][4]:.2f}</b><span>rhombohedral / hexagonal DOS from "
          f"{raman_min:g} to {raman_max:g} {CM1}</span></div>",
          f"<div><b>{cross:.0f}</b><span>Raman shift in {CM1} at which the two DOS curves cross</span></div>",
          f"<div><b>{e_lo:.0f}–{e_hi:.0f} meV</b><span>{DE} probed by the spectrometer, "
          r"\(\hbar\omega/2\)</span></div>",
          f"<div><b>{fermi['abc'][2] / fermi['ab'][2]:.3f}</b><span>rhombohedral / hexagonal DOS at the Fermi level</span></div>",
          "</div>"]

    s.append("<h2>From the ERS signal to the density of states</h2>")
    s.append(
        "<p>In graphene, few-layer graphene and graphite the ERS continuum consists of interband electron–hole "
        "excitations with negligible momentum transfer. They are created through a virtual intermediate state, appear "
        r"in crossed polarisation (\(A_{2g}\) symmetry), and the matrix element is nearly independent of energy "
        "(Kashuba and Fal'ko 2009; Ponosov et al. 2015). Ponosov et al. model the spectrum of bulk graphite with a "
        "constant matrix element. In the limit of small damping their Eq. (1) reads</p>"
        r"\[I(\omega) \propto \frac{1}{1 - e^{-\hbar\omega/k_B T}} \sum_{c,v} \int d^3p\,"
        r"\left[f(\varepsilon_v) - f(\varepsilon_c)\right]"
        r"\delta\!\left(\hbar\omega - \varepsilon_c(\mathbf{p}) + \varepsilon_v(\mathbf{p})\right).\]"
        "<p>García-Ruiz et al. (2019), Eq. (5), find that the Raman vertex is proportional to "
        r"\((\mathbf{l} \times \mathbf{l}')_z\,\sigma_z\), "
        f"so the excitations that count connect each valence band to its mirror conduction band, {PAIR}, "
        "and the spectral density reflects the joint density of states of electrons and holes. For bands that are mirror "
        r"images about the Fermi level, \(\varepsilon_c = -\varepsilon_v\), the joint density of states at "
        r"\(\hbar\omega\) is half the density of states at \(\hbar\omega/2\). This gives the working rule of this report:</p>"
        r"\[I_\mathrm{ERS}(\omega) \propto D\!\left(E_F \pm \frac{\hbar\omega}{2}\right).\]"
        r"<p>A Raman shift \(\omega\) therefore reads the DOS at an energy \(\hbar\omega/2\) from the Fermi level, and "
        f"{raman_min:g}–{raman_max:g} {CM1} ({raman_min * EV_PER_INVERSE_CM * 1e3:.1f}–"
        f"{raman_max * EV_PER_INVERSE_CM * 1e3:.0f} meV) corresponds to {DE} = {e_lo:.1f}–{e_hi:.0f} meV. "
        "The same reading underlies the layer-number assignment of Pálinkás et al. (2024), where the ERS peaks of "
        r"rhombohedral films sit at the separation of the \(-n\) and \(+n\) DOS peaks.</p>")

    s.append("<h2>Model</h2>")
    s.append(
        "<p>Both crystals are described by the Slonczewski–Weiss–McClure tight-binding model of this repository "
        "(<code>multilayer_graphene.py</code>, parameters of McEllistrim et al. 2023) extended to an infinite stack "
        r"(<code>bulk_graphite.py</code>): every hop from layer \(n\) to layer \(n + m\) carries the phase "
        r"\(e^{i m k_z d}\), "
        rf"\(d = {info['d_layer']}\) Å. The two polytypes use the same couplings and have the same atomic density, "
        f"{_sci(info['atoms_per_cm3'], 3)} {CM3}, so their DOS can be compared per atom or per volume.</p>")
    s.append('<div class="words">' + html_table(
        ["coupling", "value (eV)", "role"],
        [(r"\(\gamma_0\)", f"{par['gamma0']:.4f}", r"intralayer nearest neighbour (\(v = 1.0\times10^{6}\) m/s)"),
         (r"\(\gamma_1\)", f"{par['gamma1']:.3f}", "vertical dimer coupling"),
         (r"\(\gamma_2\)", _signed(f"{par['gamma2']:.3f}"), "next-nearest layer, through a hexagon centre"),
         (r"\(\gamma_3\)", f"{par['gamma3']:.4f}", "skew, non-dimer sites (trigonal warping)"),
         (r"\(\gamma_4\)", f"{par['gamma4']:.4f}", "skew, dimer to non-dimer (electron–hole asymmetry)"),
         (r"\(\gamma_5\)", f"{par['gamma5']:.3f}", "next-nearest layer, along a dimer chain"),
         (r"\(\Delta'\)", f"{par['Delta_prime']:.3f}", r"on-site energy per \(\gamma_1\) bond")],
        "Couplings, identical for the two polytypes") + "</div>")
    s.append(
        "<ul>"
        "<li><b>Hexagonal graphite.</b> Two layers per cell. On the K–H line the model gives the SWMcC energies"
        r"\[E_{1,2} = 2\Delta' - \gamma_5 + \tfrac{1}{2}\gamma_5\Gamma^2 \pm \gamma_1\Gamma, \qquad"
        r" E_3 = -\gamma_2 + \tfrac{1}{2}\gamma_2\Gamma^2, \qquad \Gamma = 2\cos(k_z d),\]"
        r"that is the standard model with \(\Delta = 2\Delta' + \gamma_2 - \gamma_5 ="
        rf" {swmcc_delta * 1e3:.0f}\) meV. "
        r"At each \(k_z\) the bands are those of a bilayer with coupling \(\gamma_1\Gamma\), which leaves a finite DOS "
        "at the Fermi level.</li>"
        r"<li><b>Rhombohedral graphite.</b> One layer per primitive cell, every atom has one \(\gamma_1\) bond. "
        r"The bands are Dirac cones whose apex moves on a spiral of radius \(\gamma_1/\hbar v\) around the K–H line "
        "(McClure 1969). To first order in the small couplings the apex energy is"
        r"\[E_D(k_z) \simeq \Delta' + \frac{2\gamma_1\gamma_4}{\gamma_0}"
        r" + \frac{2\gamma_4}{\gamma_0}\left(\frac{\gamma_1\gamma_3}{\gamma_0} + \frac{\gamma_2}{2}\right)\cos(3k_z d),\]"
        f"a mean of {dirac * 1e3:.1f} meV and a total width of {overlap * 1e3:.1f} meV, which is the band overlap of "
        "the semimetal.</li>"
        "</ul>")

    s.append("<h2>Density of states of the two polytypes</h2>")
    s.append(f"<figure>{img_tag(fig_comparison, 'DOS of hexagonal and rhombohedral bulk graphite versus energy')}"
             "<figcaption>Figure 1. DOS of hexagonal (blue) and rhombohedral (orange) bulk graphite, energies measured "
             "from the Fermi level of each crystal. Inset: the 100 meV around the Fermi level. Spin and valley "
             "degeneracy included; the right axis gives the same curves per volume.</figcaption></figure>")
    s.append(html_table(
        ["", "hexagonal (AB)", "rhombohedral (ABC)"],
        [("Fermi level on the model's energy scale (meV)", _signed(f"{fermi['ab'][0] * 1e3:+.1f}"),
          _signed(f"{fermi['abc'][0] * 1e3:+.1f}")),
         (rf"electrons = holes at \(T = 0\) ({CM3})", _sci(fermi["ab"][1]), _sci(fermi["abc"][1])),
         (f"DOS at the Fermi level (states {PER_EV_ATOM})", _sci(fermi["ab"][2]), _sci(fermi["abc"][2]))],
        "Fermi-level quantities from charge neutrality"))
    s.append(
        "<ul>"
        "<li>Hexagonal graphite has a finite DOS at the Fermi level and a slow, nearly linear rise on both sides. "
        f"The structure within about 25 meV of {EF} comes from the low-energy bands along K–H, whose energy "
        r"varies by \(2|\gamma_2|\) between K and H and which \(\gamma_3\) warps trigonally; the kinks near "
        r"\(-0.69\) and \(+0.87\) eV are the edges of the split bands at K (\(\pm 2\gamma_1\)).</li>"
        "<li>Rhombohedral graphite has the linear DOS of graphene, per layer, and it vanishes at the Fermi level to "
        f"within the {overlap * 1e3:.1f} meV band overlap.</li>"
        "<li>The hole side of hexagonal graphite carries 25–40 % more states than the electron side "
        r"(\(\gamma_4\) and \(\Delta'\)), while "
        r"rhombohedral graphite is symmetric to 0.2 %. The two curves therefore cross at \(+0.21\) eV for electrons "
        "and do not cross for holes within 1 eV.</li>"
        "</ul>")

    s.append("<h2>Energies accessible to the Raman spectrometer</h2>")
    s.append(f"<figure>{img_tag(fig_window, 'DOS with the energy range of the Raman spectrometer shaded')}"
             f"<figcaption>Figure 2. The same DOS with the energies reached by Raman shifts of {raman_min:g}–{raman_max:g} "
             f"{CM1} shaded: {DE} = {e_lo:.0f}–{e_hi:.0f} meV on both sides of the Fermi level. "
             f"The top axis gives the Raman shift of the {PAIR} excitation that connects "
             r"\(-E\) and \(+E\).</figcaption></figure>")
    s.append(html_table(
        [f"Raman shift ({CM1})", f"{DE} (meV)", "hexagonal", "rhombohedral", "rhombohedral / hexagonal"],
        [(f"{w:.0f}", f"{e:.1f}", f"{100 * h:.3f}", f"{100 * r:.3f}", f"{q:.2f}") for w, e, h, r, q in rows],
        r"DOS at \(E_F \mp \hbar\omega/2\), averaged over the two signs, in \(10^{-2}\) states " + PER_EV_ATOM))
    s.append(
        "<ul>"
        f"<li>Across the window the hexagonal DOS grows by a factor of {rows[-1][2] / rows[0][2]:.1f}, the rhombohedral "
        f"DOS by a factor of {rows[-1][3] / rows[0][3]:.0f}.</li>"
        f"<li>Below about 1000 {CM1} the rhombohedral crystal has less than half the DOS of the hexagonal "
        f"one; the ratio is {rows[0][4]:.2f} at {raman_min:g} {CM1}.</li>"
        f"<li>The curves meet at {cross:.0f} {CM1}; above it the rhombohedral DOS is the larger one, by "
        f"{100 * (rows[-1][4] - 1):.0f} % at {raman_max:g} {CM1}.</li>"
        "</ul>")

    s.append("<h2>What the estimate leaves out</h2>")
    s.append(
        "<ul>"
        "<li><b>Thermal factors.</b> For mirror bands the occupation factor of Eq. (1) is "
        r"\(\tanh(\hbar\omega/4k_B T)\), and the Stokes prefactor is \(\left[1 - e^{-\hbar\omega/k_B T}\right]^{-1}\). "
        r"Their product rises from \(1/4\) at \(\omega \to 0\) to 1 above roughly \(\hbar\omega = 8k_B T\) "
        f"(1700 {CM1} at 300 K). It multiplies both polytypes alike, so the ratios above are unaffected.</li>"
        f"<li><b>Spectral shape at large shifts.</b> Ponosov et al. measure a broad maximum near 3000 {CM1} in "
        "graphite and in graphene, where the joint density of states keeps rising, and discuss a phonon-assisted channel "
        "as a possible reason. The DOS is a safer guide to the ratio of the two polytypes than to the absolute line shape.</li>"
        r"<li><b>Mirror symmetry.</b> The rule \(D(E_F \pm \hbar\omega/2)\) assumes "
        r"\(\varepsilon_c = -\varepsilon_v\). The table "
        "averages the electron and hole sides, which differ by 25–40 % in hexagonal graphite inside the window.</li>"
        "<li><b>Fermi surface of hexagonal graphite.</b> Electron and hole pockets block the excitations below a few "
        f"tens of meV at low temperature; at room temperature Ponosov et al. find no gap down to 50 {CM1}.</li>"
        "<li><b>Surfaces and stacking faults.</b> Figures 1 and 2 are the bulk DOS. Finite rhombohedral films add the "
        "surface flat band and the subband edges that produce the ERS peaks of Pálinkás et al. (2024)"
        + (", shown for thick slabs below" if slabs is not None else "")
        + "; stacking faults add their own features.</li>"
        r"<li><b>Parameters.</b> McClure (1969), with \(|\gamma_4| \approx 0.25\)–\(0.28\) eV and further "
        "next-nearest-plane couplings, "
        "estimated a band overlap of 14–26 meV for rhombohedral graphite and a Fermi-level DOS of 0.07 of the hexagonal "
        rf"value. With the present couplings (\(\gamma_4 = {par['gamma4']:.3f}\) eV) the overlap is "
        f"{overlap * 1e3:.1f} meV and the "
        f"ratio {fermi['abc'][2] / fermi['ab'][2]:.3f}. The difference is confined to {DE} of the order of "
        f"the overlap, below the {e_lo:.0f} meV where the window starts.</li>"
        "</ul>")

    s.append("<h2>Numerical details and checks</h2>")
    s.append(
        "<ul>"
        rf"<li>\(k_z\) is sampled at {info['n_kz']} values over half of its period. For each value the in-plane "
        f"integral uses linear interpolation on triangles of a square mesh around K (spacing {info['dk']:g} "
        f"{PER_ANGSTROM}). Cells in "
        "which a band comes within 0.12 eV of the Fermi level are subdivided "
        f"{info['ab']['refine']} times (hexagonal) or {info['abc']['refine']} times (rhombohedral). Energy bins are "
        f"{info['bin_width'] * 1e3:g} meV wide, with no further smoothing." + ("" if not info["gaussian_sigma"] else
        f" A Gaussian of {info['gaussian_sigma'] * 1e3:g} meV was applied.") + "</li>"
        r"<li>With these default settings, halving the mesh spacing, doubling the number of \(k_z\) values or "
        "doubling the subdivision changes the hexagonal DOS by at most 0.1 % at all energies. The rhombohedral DOS "
        r"changes by less than 0.1 % for \(|E - E_F| > 3\) meV and by 0.7 % inside.</li>"
        r"<li>The Fermi level follows from charge neutrality at \(T = 0\). The carrier density of hexagonal graphite, "
        f"{_sci(fermi['ab'][1])} {CM3} of each sign, compares with the "
        r"\(25\times10^{-6}\) carriers per atom (\(2.8\times10^{18}\) "
        f"{CM3}) that McClure (1969) quotes for hexagonal graphite.</li>"
        "<li>Tests (<code>tests/test_bulk_graphite.py</code>): the K–H energies of hexagonal graphite equal the SWMcC "
        "expressions; bulk Bloch states solve the finite-stack Hamiltonian of <code>multilayer_graphene.py</code> in its "
        r"interior to \(10^{-12}\) eV; with \(\gamma_0\) and \(\gamma_1\) only, the rhombohedral DOS equals that of "
        r"graphene and the hexagonal DOS equals the \(k_z\) average of bilayers.</li>"
        "</ul>")
    s.append("<h3>Reproduce</h3>")
    s.append("<pre>python -m pytest tests -q\n"
             f"python graphite_ers_background.py --raman-min {raman_min:g} --raman-max {raman_max:g}"
             + (f" --slabs {' '.join(str(n) for n in slab_info['layers'])}" if slabs is not None else "") + "\n"
             "python graphite_ers_background.py --from-csv examples/graphite_ers_dos.csv --raman-min 50   # replot only</pre>")

    if slabs is not None:
        s.extend(_slab_section(slabs, slab_info, table, info, raman_min, raman_max, fig_slabs))
    if curves is not None:
        s.extend(_thermal_section(curves, table, info, slabs, slab_info, raman_min, raman_max, temperature, fig_thermal))

    s.append("<h2>References</h2>")
    s.append(
        "<ul>"
        "<li>O. Kashuba and V. I. Fal'ko, Signature of electronic excitations in the Raman spectrum of graphene, "
        "Phys. Rev. B 80, 241404 (2009).</li>"
        '<li>Yu. S. Ponosov, A. V. Ushakov and S. V. Streltsov, Electronic Raman scattering in graphite and single-layer '
        'and few-layer graphene, Phys. Rev. B 91, 195435 (2015), <a href="https://doi.org/10.1103/PhysRevB.91.195435">'
        "doi:10.1103/PhysRevB.91.195435</a>.</li>"
        "<li>A. García-Ruiz, S. Slizovskiy, M. Mucha-Kruczyński and V. I. Fal'ko, Spectroscopic signatures of electronic "
        'excitations in Raman scattering in thin films of rhombohedral graphite, Nano Lett. 19, 6152 (2019), '
        '<a href="https://doi.org/10.1021/acs.nanolett.9b02196">doi:10.1021/acs.nanolett.9b02196</a>.</li>'
        "<li>A. Pálinkás et al., Identification of graphite with perfect rhombohedral stacking by electronic Raman "
        'scattering, Carbon 230, 119608 (2024), <a href="https://doi.org/10.1016/j.carbon.2024.119608">'
        "doi:10.1016/j.carbon.2024.119608</a>.</li>"
        "<li>J. W. McClure, Electron energy band structure and electronic properties of rhombohedral graphite, "
        "Carbon 7, 425 (1969).</li>"
        "<li>A. McEllistrim, A. Garcia-Ruiz, Z. A. H. Goodwin and V. I. Fal'ko, Spectroscopic signatures of tetralayer "
        "graphene polytypes, Phys. Rev. B 107, 155147 (2023), arXiv:2302.07374.</li>"
        "</ul>")
    s.append("</div></body></html>" if standalone else "</div>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(s))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raman-min", type=float, default=80.0, help="lowest Raman shift of the instrument (1/cm)")
    ap.add_argument("--raman-max", type=float, default=6200.0, help="highest Raman shift of the instrument (1/cm)")
    ap.add_argument("--prefix", default="examples/graphite_ers_dos", help="path prefix of the output files")
    ap.add_argument("--from-csv", default=None, help="replot a table written by an earlier run")
    ap.add_argument("--n-kz", type=int, default=240, help="k_z values in half of the k_z period")
    ap.add_argument("--dk", type=float, default=2.0e-3, help="coarse in-plane mesh spacing (1/A)")
    ap.add_argument("--refine", type=int, default=8,
                    help="mesh subdivision near the Fermi level (hexagonal; four times this for rhombohedral)")
    ap.add_argument("--sigma", type=float, default=0.0, help="Gaussian smoothing of the DOS (eV)")
    ap.add_argument("--slabs", type=int, nargs="*", default=[20, 50, 100],
                    help="layer numbers of the finite slabs (none: skip the slabs); a slab table "
                         "next to --from-csv is reused")
    ap.add_argument("--slab-dk", type=float, default=1.0e-3, help="in-plane mesh spacing for the slabs (1/A)")
    ap.add_argument("--temperature", type=float, default=300.0,
                    help="temperature (K) of the thermally broadened curves (0: skip them)")
    ap.add_argument("--page-only", default=None, metavar="PATH",
                    help="also write the report without its document skeleton, for a publisher that adds its own")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")

    if args.from_csv:
        table, info = load_csv(args.from_csv)
    else:
        mlg.use_paper_parameters()
        table, info = compute(n_kz=args.n_kz, dk=args.dk, refine=args.refine, sigma=args.sigma)
        save_csv(args.prefix + ".csv", table, info)
        print(f"table written to {args.prefix}.csv")
    slabs = slab_info = fig_slabs = None
    slab_csv = (args.from_csv[:-len(".csv")] if args.from_csv else args.prefix) + "_slabs.csv"
    if args.from_csv and os.path.exists(slab_csv):
        slabs, slab_info = load_slabs_csv(slab_csv)
    elif args.slabs:
        mlg.set_parameters(**info["parameters"])
        slabs, slab_info = compute_slabs(args.slabs, dk=args.slab_dk)
        save_slabs_csv(slab_csv, slabs, slab_info)
        print(f"slab table written to {slab_csv}")
    report(table, info, args.raman_min, args.raman_max)
    fig_comparison, fig_window = args.prefix + "_comparison.png", args.prefix + "_raman_window.png"
    plot_comparison(table, info, fig_comparison)
    plot_raman_window(table, info, fig_window, args.raman_min, args.raman_max)
    print(f"figures written to {fig_comparison} and {fig_window}")
    if slabs is not None:
        fig_slabs = args.prefix + "_slabs.png"
        plot_slabs(slabs, slab_info, table, info, fig_slabs, args.raman_min, args.raman_max)
        print(f"slab figure written to {fig_slabs}")
    curves = fig_thermal = None
    if args.temperature > 0.0:
        curves = thermal_curves(table, info, slabs, slab_info, args.temperature)
        fig_thermal = args.prefix + "_thermal.png"
        plot_thermal(curves, table, info, slab_info, fig_thermal, args.raman_min, args.raman_max, args.temperature)
        print(f"thermally broadened figure written to {fig_thermal}")
    write_report(args.prefix + "_report.html", table, info, args.raman_min, args.raman_max, fig_comparison, fig_window,
                 slabs, slab_info, fig_slabs, curves, args.temperature, fig_thermal)
    print(f"report written to {args.prefix}_report.html")
    if args.page_only:
        write_report(args.page_only, table, info, args.raman_min, args.raman_max, fig_comparison, fig_window,
                     slabs, slab_info, fig_slabs, curves, args.temperature, fig_thermal, standalone=False)
        print(f"page without skeleton written to {args.page_only}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
