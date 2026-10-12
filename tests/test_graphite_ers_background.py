"""
Tests of the bookkeeping in graphite_ers_background.py: Fermi level of a slab,
Raman shift to energy mapping and the tables written to disk.

Run with:  python -m pytest tests -q
"""
import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multilayer_graphene as mlg  # noqa: E402
import graphite_ers_background as ers  # noqa: E402


@pytest.fixture(autouse=True)
def paper_parameters():
    mlg.use_paper_parameters()
    yield
    mlg.use_paper_parameters()


def test_slab_dos_is_per_atom_and_centred_on_the_neutrality_point():
    # bilayer with gamma0 and gamma1 only: electron-hole symmetric, DOS known in closed form
    params = {**mlg.get_parameters(), "gamma2": 0.0, "gamma3": 0.0, "gamma4": 0.0, "gamma5": 0.0,
              "Delta_prime": 0.0, "E0": 0.03}
    energy = np.arange(-0.3, 0.3005, 1.0e-3)
    dos, fermi = ers._slab_dos((2, "ab", energy, 2.0e-3, 0.2, params))
    assert abs(fermi - 0.03) < 1e-3                       # the rigid offset E0 is found and removed
    hv = 1.5 * (params["a"] / np.sqrt(3.0)) * params["gamma0"]
    analytic = (np.abs(energy) + 0.5 * params["gamma1"]) * 2.0 / (np.pi * hv ** 2) * mlg.unit_cell_area() / 4.0
    sel = np.abs(energy) > 0.03
    assert np.abs(dos[sel] / analytic[sel] - 1.0).max() < 0.03


def test_raman_shift_reads_the_dos_at_half_the_energy():
    energy = np.linspace(-1.0, 1.0, 2001)
    table = {"energy": energy, "ab": 0.5 + np.abs(energy), "abc": 3.0 * np.abs(energy)}
    shift = 0.2 / ers.EV_PER_INVERSE_CM                   # hbar w = 0.2 eV -> |E - E_F| = 0.1 eV
    hexagonal, rhombohedral = ers.dos_at_shift(table, shift)
    assert abs(hexagonal - 0.6) < 1e-9 and abs(rhombohedral - 0.3) < 1e-9
    # the curves cross where 0.5 + E = 3 E, at E = 0.25 eV
    assert abs(ers.crossing_shift(table) * ers.EV_PER_INVERSE_CM - 0.5) < 2.0 * 10.0 * ers.EV_PER_INVERSE_CM
    rows = ers.shift_rows(table, 80.0, 6200.0)
    assert rows[0][0] == 80.0 and rows[-1][0] == 6200.0
    assert abs(ers.window_mean(energy, table["abc"], 80.0, 6200.0)
               - 3.0 * 0.25 * (80.0 + 6200.0) * ers.EV_PER_INVERSE_CM) < 2e-3


def test_tables_survive_a_round_trip(tmp_path):
    energy = np.linspace(-0.1, 0.1, 21)
    table = {"energy": energy, "ab": 1.0 + energy, "abc": np.abs(energy)}
    info = {"parameters": mlg.get_parameters(), "ab": {"fermi_energy_eV": -0.003}}
    ers.save_csv(str(tmp_path / "bulk.csv"), table, info)
    loaded, loaded_info = ers.load_csv(str(tmp_path / "bulk.csv"))
    assert all(np.allclose(loaded[k], table[k]) for k in table)
    assert loaded_info["ab"]["fermi_energy_eV"] == -0.003

    slabs = {"energy": energy, "ab_20": 1.0 + energy, "abc_20": np.abs(energy)}
    slab_info = {"layers": [20], "fermi_energy_eV": {"ab_20": -0.0026, "abc_20": 0.019}}
    ers.save_slabs_csv(str(tmp_path / "slabs.csv"), slabs, slab_info)
    loaded, loaded_info = ers.load_slabs_csv(str(tmp_path / "slabs.csv"))
    assert list(loaded) == list(slabs)
    assert all(np.allclose(loaded[k], slabs[k]) for k in slabs)
    assert loaded_info == slab_info


def test_subband_peaks_of_rhombohedral_films():
    # Garcia-Ruiz et al. (2019), Eq. (1); Palinkas et al. (2024) measure 2299 +- 40 1/cm for 12 layers
    first = ers.ers_peak(1, 12, mlg.gamma1) / ers.EV_PER_INVERSE_CM
    assert abs(first / 2299.0 - 1.0) < 0.04
    energy = np.arange(-0.3, 0.3005, 1.0e-3)
    dos = np.exp(-0.5 * ((energy + 0.071) / 0.003) ** 2) + np.exp(-0.5 * ((energy - 0.106) / 0.003) ** 2)
    assert abs(ers.subband_peak_pair(energy, dos, 0.023, 0.179, 0.015) - 0.177) < 1e-9


def test_bulk_curve_is_drawn_at_the_energy_of_the_slab():
    energy = np.linspace(-0.2, 0.2, 401)
    table = {"energy": energy, "abc": np.abs(energy)}
    info = {"abc": {"fermi_energy_eV": 0.042}}
    slabs = {"energy": energy}
    slab_info = {"fermi_energy_eV": {"abc_20": 0.019}}
    bulk = ers.bulk_on_slab_axis(slabs, slab_info, table, info, "abc", 20)
    assert abs(energy[np.argmin(bulk)] - 0.023) < 1e-9      # the bulk Dirac point, 23 meV above the slab's E_F


def test_thermal_broadening_is_the_fermi_derivative_kernel():
    energy = np.arange(-1.0, 1.0005, 1.0e-3)
    kT = ers.K_BOLTZMANN * 300.0
    inner = np.abs(energy) < 0.5
    # a constant and a straight line are left alone, a V is rounded to 2 ln(2) kT at its apex
    assert np.allclose(ers.thermal_broadening(energy, np.full_like(energy, 0.7), 300.0), 0.7)
    assert np.abs(ers.thermal_broadening(energy, 2.0 + energy, 300.0) - (2.0 + energy))[inner].max() < 1e-9
    apex = ers.thermal_broadening(energy, np.abs(energy), 300.0)[np.argmin(np.abs(energy))]
    assert abs(apex - 2.0 * np.log(2.0) * kT) < 1e-5
    # a narrow line becomes sech^2(E / 2kT) / 4kT, of full width 3.53 kT at half maximum
    line = np.where(np.abs(energy) < 1e-9, 1.0e3, 0.0)                 # unit weight in one 1 meV bin
    shape = ers.thermal_broadening(energy, line, 300.0)
    assert abs(shape.sum() * 1.0e-3 - 1.0) < 1e-9
    assert abs(shape.max() - 1.0 / (4.0 * kT)) < 2e-3 / (4.0 * kT)
    above = energy[shape >= 0.5 * shape.max()]
    assert abs((above[-1] - above[0]) - 3.5255 * kT) < 2.5e-3


def _synthetic_results():
    energy = np.arange(-1.0, 1.0005, 1.0e-3)
    table = {"energy": energy, "ab": 5.0e-3 + 1.2e-2 * np.abs(energy), "abc": 3.9e-2 * np.abs(energy)}
    info = {"parameters": mlg.get_parameters(), "d_layer": 3.35, "n_kz": 240, "dk": 2.0e-3, "bin_width": 1.0e-3,
            "gaussian_sigma": 0.0, "atoms_per_cm3": 1.139e23,
            "ab": {"fermi_energy_eV": -0.003, "carrier_density_cm3": 2.0e18, "refine": 8},
            "abc": {"fermi_energy_eV": 0.042, "carrier_density_cm3": 1.2e15, "refine": 32}}
    e = np.arange(-0.5, 0.5005, 1.0e-3)
    slabs = {"energy": e, "ab_20": 5.0e-3 + 1.2e-2 * np.abs(e),
             "abc_20": 3.9e-2 * np.abs(e - 0.023) + 8.0e-3 * np.exp(-0.5 * (e / 0.01) ** 2)}
    slab_info = {"layers": [20], "dk": 1.0e-3, "k_max": 0.18, "bin_width": 1.0e-3,
                 "fermi_energy_eV": {"ab_20": -0.003, "abc_20": 0.019}}
    return table, info, slabs, slab_info


def test_slab_curves_are_continued_by_the_bulk_before_broadening():
    table, info, slabs, slab_info = _synthetic_results()
    curves = ers.thermal_curves(table, info, slabs, slab_info, 300.0)
    assert list(curves) == ["bulk", 20]
    edge = np.abs(table["energy"] - 0.5) < 1e-9
    # the synthetic hexagonal slab equals the bulk, so its broadened curve has no step where its own data end
    assert abs(curves[20]["ab"][edge][0] - curves["bulk"]["ab"][edge][0]) < 1e-12
    # the surface band of the rhombohedral slab survives as extra states at the Fermi level
    centre = np.argmin(np.abs(table["energy"]))
    reference = ers.bulk_on_broadened_slab_axis(curves, table, info, slab_info, "abc", 20)
    assert curves[20]["abc"][centre] > 1.2 * reference[centre]


def test_report_is_complete_and_written_in_latex(tmp_path):
    table, info, slabs, slab_info = _synthetic_results()
    figures = [str(tmp_path / f"{name}.png") for name in ("comparison", "window", "slabs", "thermal")]
    curves = ers.thermal_curves(table, info, slabs, slab_info, 300.0)
    ers.plot_comparison(table, info, figures[0])
    ers.plot_raman_window(table, info, figures[1], 80.0, 6200.0)
    ers.plot_slabs(slabs, slab_info, table, info, figures[2], 80.0, 6200.0)
    ers.plot_thermal(curves, table, info, slab_info, figures[3], 80.0, 6200.0, 300.0)
    path = tmp_path / "report.html"
    ers.write_report(str(path), table, info, 80.0, 6200.0, figures[0], figures[1],
                     slabs, slab_info, figures[2], curves, 300.0, figures[3])
    page = path.read_text(encoding="utf-8")
    assert "mathjax" in page and page.count("<img ") == 4
    assert page.count("\\(") == page.count("\\)") > 50 and page.count("\\[") == page.count("\\]") >= 5
    assert "<sub>" not in page and "<sup>" not in page
    sections = [page.index(title) for title in ("<h2>Finite slabs", "<h2>Thermal broadening at 300 K", "<h2>References")]
    assert sections == sorted(sections)
