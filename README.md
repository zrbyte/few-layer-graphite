# Multilayer Graphene Band Structure Calculator

Calculating the bands of few layer graphite in a TB model, beyond the nearest neighbour.
**THIS IS JUST FOR VISUALIZATION PURPOSES!** The parameters were not fitted to ab initio, or checked regarding the final structure of the low energy bands. 

## Simple example

```python
import multilayer_graphene as mlg

# Tell it what you want
mlg.set_parameters(N_layers=3, stacking='abc')

# Plot some beautiful bands
mlg.plot_bands()

# Want to see how different layer numbers compare? No problem!
mlg.plot_panel_comparison(range(1, 9))
```

## Settings

### Hoppings in eV
Don't worry, the defaults are sensible! But if you're feeling adventurous:

- `gamma0`: How carbons talk within a layer (default: 3.0)
- `E0`: Energy offset (default: 0)
- `Delta`: A vs B asymmetry (default: 0)

### The System Stuff
- `N_layers`: How many layers? (default: 3)
- `stacking`: 'abc' for rhombohedral or 'aba' for Bernal (default: 'abc')

### The k-Space Details
- `K_point`: Where's the K point? (default: [1/3, 1/3])
`- `dk`: Legacy range around K (kept for API compatibility)
- `n_k`: How many k-points along the default path (default: 1500)
- `d_cc`: Carbon-carbon bond length in Å (default: 1.42)

By default, band plots now use the high-symmetry path Γ → K → M, and the x-axis is the arc length along this path with the K point centered at 0 (Γ at negative x, M at positive x). You can override x-limits via the `xlim` argument.

### Movie Generation
- `make_movie.py [framerate] [stacking_type] [max_layers]`: Creates both WebM video and animated GIF for maximum compatibility.
  - `python make_movie.py 2 abc` - 2 fps, ABC stacking, 30 layers
  - `python make_movie.py 5 aba 20` - 5 fps, ABA stacking, 20 layers
  - `python make_movie.py 1 abc 5` - Quick test with 5 layers
  - Requires: ffmpeg (for WebM) and Pillow (for GIF)

## What Can You Do With It?

### Set Things Up
- `set_parameters(**kwargs)`: Change whatever you want
- `get_info()`: See what you've got configured  
- `get_parameters()`: Get all the current settings

### Calculate
- `calculate_bands(...)`

### Plotting
- `plot_bands(...)`: Single band structure plot
- `plot_panel_comparison(...)`: Side-by-side comparison panels
- `plot_band_density(...)`: Density plot of a selected band.

### Brillouin Zone helpers
- `get_brillouin_zone(plot=True, ax=None, annotate=True, figsize=(4,4), save_as=None, show=True)`: Returns a dictionary with reciprocal vectors (`b1`, `b2`), Γ, K and M points (both fractional and Cartesian), and the first Brillouin zone hexagon vertices (Å⁻¹). If `plot=True`, it also draws the BZ hexagon and marks Γ, K, and M; you can pass an existing Matplotlib `ax` to draw into, or let it create one.


Based on the work from Grüneiss et al. (Physical Review B 78, 205425, 2008).

```python
mlg.set_parameters(N_layers=N, stacking='abc', **params)
E, k_mag = mlg.calculate_bands()

# Same deal for ABA:
mlg.set_parameters(N_layers=N, stacking='aba', **params)
E, k_mag = mlg.calculate_bands()
```

## What You Need

- NumPy (for the math)
- Matplotlib (for the pretty plots)
