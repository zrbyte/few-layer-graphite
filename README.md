# Few-layer graphene: SWMcC tight-binding bands and density of states

`multilayer_graphene.py` computes band structures and densities of states (DOS)
of N-layer graphene with any stacking sequence: rhombohedral (ABC…), Bernal
(ABAB…) and mixed polytypes (ABCB, ABAC, ABCAB, …).  The model is the full
Slonczewski–Weiss–McClure (SWMcC) tight-binding Hamiltonian with the couplings
γ0 … γ5 and the dimer on-site energy Δ'.  Conventions and default parameters follow

> A. McEllistrim, A. Garcia-Ruiz, Z. A. H. Goodwin, V. I. Fal'ko,
> *Spectroscopic signatures of tetralayer graphene polytypes*, arXiv:2302.07374 (2023)

whose hybrid k·p Hamiltonians (their Eq. 1) are the first-order expansion of
this lattice model around the valley centres.  Both models are implemented
(`hamiltonian`, `kp_hamiltonian`) and share one assembly routine, so the
correspondence is exact by construction and is checked by the test suite.

## Model

Basis per layer: (A_n, B_n), B_n displaced by τ = (0, a/√3) from A_n, a = 2.46 Å.
Following the cyclic sequence A→B→C→A ("forward" step) puts B_{n+1} on top of
A_n (dimer A_n–B_{n+1}); a "backward" step (B→A, C→B, A→C) puts A_{n+1} on top
of B_n.  For each adjacent pair the block is the paper's V_AB (forward) or
V_AB† (backward); for next-nearest layers the block is W_ABC, W_ABC†, W_ABA or
W_BAB depending on the two steps in between; each site receives Δ' times the
number of γ1 bonds attached to it.  This reproduces Eq. (1a)–(1d) of the paper
and extends them to any sequence.  (Two misprints of arXiv v1 are corrected:
the layer-3 on-site term of H_ABCB must be Δ'(1 − σ_z) and the (2,3) block of
H_ABAC must be V_AB†; the band-edge labels of the paper's Fig. 1, e.g.
√2γ1 + 3Δ'/2 for ABCB, require this.)

| coupling | meaning | matrix element | default (eV) |
|---|---|---|---|
| γ0 | intralayer nearest neighbour | −γ0 f(k) | 3.0896 (v = 1.0×10⁶ m/s) |
| γ1 | vertical dimer, adjacent layers | γ1 | 0.390 |
| γ3 | skew, non-dimer sites, adjacent layers | γ3 f(k) | 0.3090 (v3 = 0.1×10⁶ m/s) |
| γ4 | skew, dimer–non-dimer, adjacent layers | γ4 f*(k) | 0.0680 (v4 = 0.022×10⁶ m/s) |
| γ2 | vertical, two layers apart, through a hexagon centre | γ2/2 | −0.017 |
| γ5 | vertical, two layers apart, along a dimer chain | γ5/2 | 0.038 |
| Δ' | on-site energy per γ1 bond | Δ' × (bonds) | 0.025 |

f(k) = Σ_j exp(i k·δ_j) over the three A→B bond vectors; near K_ξ,
−γ0 f = ħv (ξ p_x − i p_y), with ħv_i = (3/2) a_cc γ_i.  Velocities can be
given directly: `set_parameters(v=1.0e6, v3=0.1e6, v4=0.022e6)`.

## Usage

```python
import multilayer_graphene as mlg

mlg.set_parameters(N_layers=4, stacking='abc')        # periodic pattern, or
mlg.set_parameters(stacking='ABCB')                   # explicit sequence

E, k = mlg.calculate_bands(path_type='px')            # bands along p_x through K (1/Å)
mlg.plot_bands(E, k, path_type='px', xlim=(-0.2, 0.2), ylim=(-0.8, 0.8))

dos = mlg.calculate_density_of_states((-0.8, 0.8))    # dict: energy, dos_per_cm2, ...
mlg.plot_dos((-0.8, 0.8), units='cm2')                # same layout as Fig. 1 of the paper

H = mlg.hamiltonian(kx, ky, stacking_type='ABCA')     # (…, 2N, 2N) for array kx, ky (1/Å)
H_kp = mlg.kp_hamiltonian(px, py, stacking_type='ABCA', xi=+1)
```

Paths: `'gkm'` (Γ→K→M, default), `'gkg'`, `'px'`, `'py'`; K is always at
coordinate 0.  `plot_panel_comparison(range(1, 9))` draws several layer numbers
side by side, `plot_band_density(b)` maps one band around K.

### Density of states

`calculate_density_of_states` integrates the bands with the linear-interpolation
(triangle) method on a square k-mesh around K (`region='valley'`, spacing `dk`,
valley degeneracy 2 applied) or over the full Brillouin zone (`region='bz'`).
Results are returned per unit cell, per Å² and per cm² (spin degeneracy 2
included).  `model='kp'` evaluates the paper's continuum Hamiltonian instead of
the lattice one.  `plot_dos(..., csv_path=...)` / `load_dos_from_csv` store and
reload curves.

## Comparison with McEllistrim et al. (2023)

`python compare_with_mcellistrim.py` overlays the tight-binding and k·p DOS with
the curves digitised from the paper's Fig. 1 (tetralayers, 1.8 meV per pixel)
and Fig. 7 (trilayers and pentalayers, 7.5 meV per pixel); the digitised data
live in `tests/data/`.  With the paper's parameters:

- every band edge and van Hove singularity appears at the published energy:
  local cross-correlation of the curves gives shifts below 5 meV (Fig. 1) and
  7 meV (Fig. 7); the dimer edges at Δ' ± γ1, the ABAB steps at −0.571, −0.226,
  0.256, 0.691 eV and the ABCB steps at −0.505, 0.599 eV are reproduced;
- the absolute DOS of the k·p model is within 1–3 % (electrons) and 6–9 %
  (holes) of Fig. 1 and within 1–6 % of the trilayer and pentalayer curves of
  Fig. 7; the lattice model is a further 2 % higher because its band velocity
  decreases slightly away from K.  The hole-side excess of Fig. 1 is absent in
  Fig. 7, whose curves show the stronger electron–hole asymmetry, so the residual
  reflects the numerics of the published figures rather than the parameters;
- the ABA trilayer panel of Fig. 7 is drawn on a 10¹³ scale and lies a factor
  of two below the model and below every other panel.

## Tests

```
python -m pytest tests -q
```

The tests check the parameter conversion, Hermiticity, time reversal and zone
periodicity, the exact reduction of the lattice model to the paper's Eq. (1)
at K, analytic band edges (bilayer, ABC trilayer, ABAB and ABCB dimer chains),
the DOS normalisation (monolayer, bilayer, full-zone state count) and the
agreement with the digitised DOS of the paper.

## Other scripts

- `example_usage.py`: short tour of the API.
- `make_movie.py [fps] [abc|aba] [max_layers]`: band-structure animation versus
  layer number (needs ffmpeg and/or Pillow).
- `multilayer_graphene_pythtb.py`: an older PythTB-based implementation with
  different parameters; kept for reference only, not maintained.

Requirements: NumPy, Matplotlib (pytest for the tests).
