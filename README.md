# Multilayer Graphene Band Structure Calculator

Hey there! 👋 This is a friendly Python toolkit for exploring the fascinating world of multilayer graphene band structures. Whether you're into ABC (rhombohedral) or ABA (Bernal) stacking, we've got you covered!

## What's Cool About This?

- **One-Stop Shop**: Everything you need for both ABC and ABA stacking in one place
- **Super Simple**: Just set a few global variables and you're good to go
- **Any Number of Layers**: From monolayer to... well, as many as your computer can handle!
- **Pretty Plots**: Built-in functions that make your band structures look awesome
- **Solid Physics**: Built on the tried-and-true Grüneis parameters (PRB 2008)

## Getting Started (It's Really Easy!)

```python
import multilayer_graphene as mlg

# Tell it what you want
mlg.set_parameters(N_layers=3, stacking='abc')

# Plot some beautiful bands
mlg.plot_bands()

# Want to see how different layer numbers compare? No problem!
mlg.plot_panel_comparison(range(1, 9))
```

## Tweak the Settings (If You Want To)

### The Physics Knobs (Tight-binding Parameters in eV)
Don't worry, the defaults are sensible! But if you're feeling adventurous:

- `gamma0`: How carbons talk within a layer (default: 3.053)
- `gamma1`: Vertical hopping between layers (default: 0.403)  
- `gamma2`: Next-nearest A↔A coupling (default: -0.025)
- `gamma3`: Skew hopping (default: 0.274)
- `gamma4`: Like-sublattice coupling (default: 0.143)
- `gamma5`: Next-nearest B↔B coupling (default: 0.030)
- `E0`: Energy offset (default: -0.025)
- `Delta`: A vs B asymmetry (default: -0.005)

### The System Stuff
- `N_layers`: How many layers? (default: 3)
- `stacking`: 'abc' for rhombohedral or 'aba' for Bernal (default: 'abc')

### The k-Space Details
- `K_point`: Where's the K point? (default: [1/3, 1/3])
- `dk`: How far around K to look (default: 1.5)
- `n_k`: How many k-points (default: 1500 - plenty!)
- `d_cc`: Carbon-carbon bond length in Å (default: 1.42)

## What Can You Do With It?

### Set Things Up
- `set_parameters(**kwargs)`: Change whatever you want
- `get_info()`: See what you've got configured  
- `get_parameters()`: Get all the current settings

### Calculate Stuff
- `calculate_bands(...)`: The main event - get those band structures!
- `build_hamiltonian(...)`: Build the matrix if you're into that
- `build_hamiltonian_abc(...)` / `build_hamiltonian_aba(...)`: Stacking-specific versions

### Make Pretty Pictures
- `plot_bands(...)`: Single band structure plot
- `plot_panel_comparison(...)`: Side-by-side comparison panels (very satisfying!)

### Helper Functions
- `get_k_path()`: Generate a nice k-path around the K point
- `f1(kx, ky)`: The structure factor (for the mathematically curious)

## The Science Behind It

This is all based on the solid work from Grüneiss et al. (Physical Review B 78, 205425, 2008) - they did the heavy lifting on the tight-binding parameters.

### Stacking: ABC vs ABA (What's the Difference?)

**ABC (Rhombohedral)**: Think of it like a spiral staircase - each layer rotates consistently
- Goes like: A₁B₁-A₂B₂-A₃B₃-... 
- All the layer couplings follow the same pattern (mathematically, all T₊ matrices)

**ABA (Bernal)**: More like a back-and-forth dance
- Goes like: A₁B₁-A₂B₂-A₁B₁-... 
- The coupling alternates between T₊ and T₋ matrices

### The Math (For Those Who Care)

We build a 2N×2N Hamiltonian matrix where N is the number of layers:
- **Within each layer**: H₀ = [[E_A, γ₀f], [γ₀f*, E_B]] (the usual graphene stuff)
- **Between neighboring layers**: T₊ or T₋ matrices (depends on the stacking)
- **Next-nearest layers**: Simple diagonal terms with γ₂ and γ₅

The structure factor f(k) = 1 + exp(i2πk_x) + exp(i2πk_y) captures the hexagonal lattice geometry.

## Migrating from the Old Scripts?

Had some old `nnn-bands-ABC.py` and `nnn-bands-ABA.py` scripts? Here's how to update:

```python
# The old way:
# bands_ABC(N, params)

# The new way:
mlg.set_parameters(N_layers=N, stacking='abc', **params)
E, k_mag = mlg.calculate_bands()

# Same deal for ABA:
# Old: bands_AB(N, params)  
# New: 
mlg.set_parameters(N_layers=N, stacking='aba', **params)
E, k_mag = mlg.calculate_bands()
```

## What You Need

Just the basics:
- NumPy (for the math)
- Matplotlib (for the pretty plots)
- optionally, Pythtb

## Want Even More? Try the PythTB Version!

There's also a PythTB-based version in `multilayer_graphene_pythtb.py` that does the same thing but with extra bells and whistles:

- Plugs into the whole PythTB ecosystem
- Built-in validation (catches your mistakes before you do!)
- Easy to extend to other tight-binding models

Using it is just as easy:
```python
import multilayer_graphene_pythtb as mlg_pythtb

# Same commands, same results!
mlg_pythtb.set_parameters(N_layers=3, stacking='abc')
mlg_pythtb.plot_bands()
```

Curious about the differences? Check out `compare_implementations.py` - it'll show you both approaches side by side.
