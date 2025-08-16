# Multilayer Graphene Band Structure Calculator

A unified Python module for calculating band structures of multilayer graphene with both ABC (rhombohedral) and ABA (Bernal) stacking configurations.

## Features

- **Unified Interface**: Single module combining both ABC and ABA stacking calculations
- **Global Configuration**: Easy parameter control via global variables
- **Flexible Calculations**: Calculate bands for any number of layers
- **Built-in Plotting**: Ready-to-use plotting functions for band structures
- **Grüneis Parameters**: Uses tight-binding parameters from Grüneis PRB 2008

## Quick Start

```python
import multilayer_graphene as mlg

# Set configuration
mlg.set_parameters(N_layers=3, stacking='abc')

# Calculate and plot bands
mlg.plot_bands()

# Panel comparison for different layer numbers
mlg.plot_panel_comparison(range(1, 9))
```

## Global Configuration Variables

### Tight-binding Parameters (eV)
- `gamma0`: Intralayer nearest-neighbor (default: 3.053)
- `gamma1`: Interlayer vertical dimer (default: 0.403)
- `gamma2`: Next-nearest layer A↔A (default: -0.025)
- `gamma3`: Interlayer skew coupling (default: 0.274)
- `gamma4`: Interlayer like-sublattice (default: 0.143)
- `gamma5`: Next-nearest layer B↔B (default: 0.030)
- `E0`: On-site energy shift (default: -0.025)
- `Delta`: A vs B on-site asymmetry (default: -0.005)

### System Configuration
- `N_layers`: Number of layers (default: 3)
- `stacking`: Stacking type - 'abc' or 'aba' (default: 'abc')

### k-path Configuration
- `K_point`: K point in reciprocal lattice units (default: [1/3, 1/3])
- `dk`: k-space range around K (default: 1.5)
- `n_k`: Number of k-points (default: 1500)
- `d_cc`: Carbon-carbon distance in Å (default: 1.42)

## Main Functions

### Configuration
- `set_parameters(**kwargs)`: Set global parameters
- `get_info()`: Display current configuration
- `get_parameters()`: Get parameters as dictionary

### Calculations
- `calculate_bands(N, stacking_type, k_path)`: Calculate band structure
- `build_hamiltonian(kx, ky, N, stacking_type)`: Build Hamiltonian matrix
- `build_hamiltonian_abc(kx, ky, N)`: ABC-specific Hamiltonian
- `build_hamiltonian_aba(kx, ky, N)`: ABA-specific Hamiltonian

### Plotting
- `plot_bands(...)`: Plot single band structure
- `plot_panel_comparison(...)`: Plot comparison panels

### Utilities
- `get_k_path()`: Generate k-path around K point
- `f1(kx, ky)`: Structure factor function

## Theory Background

This implementation uses the tight-binding model from:
> Grüneiss et al., Physical Review B 78, 205425 (2008)

### Stacking Types

**ABC (Rhombohedral)**: All adjacent layer couplings use the same T₊ matrix
- Layer sequence: A₁B₁-A₂B₂-A₃B₃-... with consistent rotation

**ABA (Bernal)**: Adjacent layer couplings alternate between T₊ and T₋ matrices  
- Layer sequence: A₁B₁-A₂B₂-A₁B₁-... with alternating alignment

### Hamiltonian Structure

For N layers, the Hamiltonian is 2N×2N with:
- **Intralayer blocks**: H₀ = [[E_A, γ₀f], [γ₀f*, E_B]]
- **Adjacent layers**: T₊ or T₋ matrices depending on stacking
- **Next-nearest layers**: Diagonal coupling with γ₂ (A-A) and γ₅ (B-B)

Where f(k) = 1 + exp(i2πk_x) + exp(i2πk_y) is the structure factor.

## Migration from Original Scripts

The original `nnn-bands-ABC.py` and `nnn-bands-ABA.py` functionality is now available as:

```python
# Old: bands_ABC(N, params)
# New: 
mlg.set_parameters(N_layers=N, stacking='abc', **params)
E, k_mag = mlg.calculate_bands()

# Old: bands_AB(N, params)  
# New:
mlg.set_parameters(N_layers=N, stacking='aba', **params)
E, k_mag = mlg.calculate_bands()
```

## Requirements

- NumPy
- Matplotlib

## Alternative Implementation

A PythTB-based implementation is also available in `multilayer_graphene_pythtb.py` that provides the same functionality using the PythTB tight-binding framework. This offers:

- Integration with the established PythTB ecosystem
- Built-in model validation and analysis tools
- Extensibility to other tight-binding models

To use the PythTB implementation:
```python
import multilayer_graphene_pythtb as mlg_pythtb

# Same API as the direct implementation
mlg_pythtb.set_parameters(N_layers=3, stacking='abc')
mlg_pythtb.plot_bands()
```

See `compare_implementations.py` for detailed comparison between both approaches.
