"""
Multilayer graphene band structure calculator

This module provides functions to calculate band structures for multilayer graphene
with both ABC (rhombohedral) and ABA (Bernal) stacking using Grüneis PRB 2008 parameters.

Global configuration variables control the calculation parameters:
- gamma0 to gamma5: Tight-binding parameters (eV)
- E0, Delta: On-site energies (eV)
- N_layers: Number of layers
- stacking: 'abc' or 'aba'
- k_path_config: k-point path configuration

Author: Generated from nnn-bands-ABC.py and nnn-bands-ABA.py
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# =============================================================================
# Global Configuration Variables
# =============================================================================

# Grüneis PRB 2008 (TB–GW) parameters [eV]
gamma0 = 3.053   # intralayer nearest-neighbor
gamma1 = 0.403   # interlayer vertical dimer
gamma2 = -0.025  # next-nearest layer (A<->A)
gamma3 = 0.274   # interlayer skew (A_n <-> B_{n+1})
gamma4 = 0.143   # interlayer like-sublattice (A<->A, B<->B) across adjacent layers
gamma5 = 0.030   # next-nearest layer (B<->B)
E0 = -0.025      # on-site shift
Delta = -0.005   # A vs B on-site asymmetry

# System configuration
N_layers = 3           # Number of layers
stacking = 'abc'       # 'abc' for rhombohedral or 'aba' for Bernal

# k-path configuration around K point
K_point = np.array([1/3, 1/3], float)  # K point in reciprocal lattice units
dk = 1.5              # k-space range around K
n_k = 1500            # Number of k-points
d_cc = 1.42           # Carbon-carbon distance (Å)

# =============================================================================
# Helper Functions
# =============================================================================

def get_k_path():
    """
    Generate k-path around K point in reciprocal lattice units.
    
    Returns:
        tuple: (k_path_recip, k_mag) where k_path_recip is the k-path in 
               reciprocal lattice units and k_mag is |k| for plotting (Å⁻¹)
    """
    k_path_recip = np.column_stack([
        np.linspace(K_point[0]-dk, K_point[0]+dk, n_k),
        np.full(n_k, K_point[1])
    ])
    
    # |k| for plotting (Å⁻¹)
    k_mag = (2*np.pi/d_cc) * np.sqrt(
        (4/3)*k_path_recip[:,0]**2 - (4/9)*k_path_recip[:,0] + 4/27
    )
    
    return k_path_recip, k_mag

def f1(kx_rec, ky_rec):
    """
    Structure factor for graphene near K point.
    
    Args:
        kx_rec, ky_rec: k-coordinates in reciprocal lattice units
        
    Returns:
        complex: Structure factor f(k) = 1 + exp(i2πkx) + exp(i2πky)
    """
    return 1.0 + np.exp(2j*np.pi*kx_rec) + np.exp(2j*np.pi*ky_rec)

def get_parameters():
    """
    Get current tight-binding parameters as dictionary.
    
    Returns:
        dict: Dictionary containing all gamma parameters and on-site energies
    """
    return {
        'gamma0': gamma0, 'gamma1': gamma1, 'gamma2': gamma2,
        'gamma3': gamma3, 'gamma4': gamma4, 'gamma5': gamma5,
        'E0': E0, 'Delta': Delta
    }

# =============================================================================
# Hamiltonian Construction
# =============================================================================

def build_hamiltonian_abc(kx_rec, ky_rec, N=None):
    """
    Build Hamiltonian for ABC (rhombohedral) stacked multilayer graphene.
    
    In ABC stacking, all adjacent layer couplings use T_plus matrix.
    
    Args:
        kx_rec, ky_rec: k-coordinates in reciprocal lattice units
        N: Number of layers (uses global N_layers if None)
        
    Returns:
        ndarray: (2N x 2N) Hamiltonian matrix
    """
    if N is None:
        N = N_layers
        
    f = f1(kx_rec, ky_rec)
    fc = np.conj(f)
    
    # On-site energies
    EA, EB = E0 + Delta, E0
    
    dim = 2*N
    H = np.zeros((dim, dim), dtype=complex)
    
    # Intralayer blocks: H_0 = [[EA, γ0*f], [γ0*f*, EB]]
    for n in range(N):
        iA, iB = 2*n, 2*n+1
        H[iA, iA] += EA
        H[iB, iB] += EB
        H[iA, iB] += gamma0 * f
        H[iB, iA] += gamma0 * fc
    
    # Adjacent layers: n -> n+1, always T_plus
    # T_plus = [[γ4*f, γ3*f*], [γ1, γ4*f]]
    for n in range(N-1):
        iA, iB = 2*n, 2*n+1
        jA, jB = 2*(n+1), 2*(n+1)+1
        
        H[iA, jA] += gamma4 * f
        H[iB, jA] += gamma1
        H[iA, jB] += gamma3 * fc
        H[iB, jB] += gamma4 * f
        
        # Hermitian conjugate
        H[jA, iA] = np.conj(H[iA, jA])
        H[jA, iB] = np.conj(H[iB, jA])
        H[jB, iA] = np.conj(H[iA, jB])
        H[jB, iB] = np.conj(H[iB, jB])
    
    # Next-nearest layers: n -> n+2, diagonal coupling
    for n in range(N-2):
        iA, iB = 2*n, 2*n+1
        kA, kB = 2*(n+2), 2*(n+2)+1
        H[iA, kA] += gamma2
        H[iB, kB] += gamma5
        H[kA, iA] += gamma2
        H[kB, iB] += gamma5
    
    return H

def build_hamiltonian_aba(kx_rec, ky_rec, N=None):
    """
    Build Hamiltonian for ABA (Bernal) stacked multilayer graphene.
    
    In ABA stacking, adjacent layer couplings alternate between T_plus and T_minus.
    
    Args:
        kx_rec, ky_rec: k-coordinates in reciprocal lattice units
        N: Number of layers (uses global N_layers if None)
        
    Returns:
        ndarray: (2N x 2N) Hamiltonian matrix
    """
    if N is None:
        N = N_layers
        
    f = f1(kx_rec, ky_rec)
    fc = np.conj(f)
    
    # On-site energies
    EA, EB = E0 + Delta, E0
    
    dim = 2*N
    H = np.zeros((dim, dim), dtype=complex)
    
    # Intralayer blocks: H_0 = [[EA, γ0*f], [γ0*f*, EB]]
    for n in range(N):
        iA, iB = 2*n, 2*n+1
        H[iA, iA] += EA
        H[iB, iB] += EB
        H[iA, iB] += gamma0 * f
        H[iB, iA] += gamma0 * fc
    
    # Adjacent layers: n -> n+1, alternating T_plus and T_minus
    for n in range(N-1):
        iA, iB = 2*n, 2*n+1
        jA, jB = 2*(n+1), 2*(n+1)+1
        
        if n % 2 == 0:
            # T_plus = [[γ4*f, γ3*f*], [γ1, γ4*f]]
            H[iA, jA] += gamma4 * f
            H[iB, jA] += gamma1
            H[iA, jB] += gamma3 * fc
            H[iB, jB] += gamma4 * f
        else:
            # T_minus = [[γ4*f*, γ1], [γ3*f, γ4*f*]]
            H[iA, jA] += gamma4 * fc
            H[iA, jB] += gamma1
            H[iB, jA] += gamma3 * f
            H[iB, jB] += gamma4 * fc
        
        # Hermitian conjugate
        H[jA, iA] = np.conj(H[iA, jA])
        H[jA, iB] = np.conj(H[iB, jA])
        H[jB, iA] = np.conj(H[iA, jB])
        H[jB, iB] = np.conj(H[iB, jB])
    
    # Next-nearest layers: n -> n+2, diagonal coupling
    for n in range(N-2):
        iA, iB = 2*n, 2*n+1
        kA, kB = 2*(n+2), 2*(n+2)+1
        H[iA, kA] += gamma2
        H[iB, kB] += gamma5
        H[kA, iA] += gamma2
        H[kB, iB] += gamma5
    
    return H

def build_hamiltonian(kx_rec, ky_rec, N=None, stacking_type=None):
    """
    Build Hamiltonian for multilayer graphene with specified stacking.
    
    Args:
        kx_rec, ky_rec: k-coordinates in reciprocal lattice units
        N: Number of layers (uses global N_layers if None)
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        
    Returns:
        ndarray: (2N x 2N) Hamiltonian matrix
    """
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    
    if stacking_type.lower() == 'abc':
        return build_hamiltonian_abc(kx_rec, ky_rec, N)
    elif stacking_type.lower() == 'aba':
        return build_hamiltonian_aba(kx_rec, ky_rec, N)
    else:
        raise ValueError(f"Unknown stacking type: {stacking_type}. Use 'abc' or 'aba'.")

# =============================================================================
# Band Structure Calculation
# =============================================================================

def calculate_bands(N=None, stacking_type=None, k_path=None):
    """
    Calculate band structure for multilayer graphene.
    
    Args:
        N: Number of layers (uses global N_layers if None)
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        k_path: (k_path_recip, k_mag) tuple (generates from globals if None)
        
    Returns:
        tuple: (eigenvalues, k_mag) where eigenvalues has shape (n_k, 2N)
               and k_mag is for plotting
    """
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    if k_path is None:
        k_path_recip, k_mag = get_k_path()
    else:
        k_path_recip, k_mag = k_path
    
    E = np.empty((len(k_path_recip), 2*N))
    
    for idx, (kx, ky) in enumerate(k_path_recip):
        H = build_hamiltonian(kx, ky, N, stacking_type)
        eigenvals = np.linalg.eigvalsh(H)
        E[idx] = np.sort(eigenvals.real)
    
    return E, k_mag

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_bands(E=None, k_mag=None, N=None, stacking_type=None, 
               xlim=(2.8, 3.1), ylim=(-0.5, 0.5), figsize=(5, 4),
               highlight_middle=True, save_as=None):
    """
    Plot band structure for single layer number.
    
    Args:
        E: Eigenvalues array (calculates if None)
        k_mag: k-magnitude array for x-axis (calculates if None)
        N: Number of layers (uses global N_layers if None)
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        xlim, ylim: Plot limits
        figsize: Figure size
        highlight_middle: Whether to highlight middle bands with thicker lines
        save_as: Filename to save plot (e.g., 'bands.svg', 'bands.png'). If None, just show.
    """
    if E is None or k_mag is None:
        E, k_mag = calculate_bands(N, stacking_type)
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    
    plt.figure(figsize=figsize)
    nb = E.shape[1]
    mid = [nb//2-1, nb//2] if highlight_middle else []
    
    for b in range(nb):
        lw = 0.8 if b in mid else 0.5
        plt.plot(k_mag, E[:,b], linewidth=lw, color='black')  # Plot all bands in black
    
    plt.ylabel("Energy (eV)")
    plt.title(f"{stacking_type.upper()} | Grüneis TB–GW | {N}-layer")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', dpi=300)  # High quality save with tight bbox
        print(f"Plot saved as: {save_as}")  # Confirm save
    else:
        plt.show()

def plot_panel_comparison(N_range=range(1, 9), stacking_type=None,
                         xlim=(2.8, 3.1), ylim=(-0.7, 0.7), figsize=(16, 2), save_as=None):
    """
    Plot comparison panels for different layer numbers.
    
    Args:
        N_range: Range of layer numbers to plot
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        xlim, ylim: Plot limits
        figsize: Figure size
        save_as: Filename to save plot (e.g., 'comparison.svg', 'comparison.png'). If None, just show.
    """
    if stacking_type is None:
        stacking_type = stacking
    
    # Calculate all band structures
    k_path_recip, k_mag = get_k_path()
    bandsets = {}
    for N in N_range:
        E, _ = calculate_bands(N, stacking_type, (k_path_recip, k_mag))
        bandsets[N] = E
    
    # Create panels
    fig, axes = plt.subplots(1, len(N_range), figsize=figsize)
    if len(N_range) == 1:
        axes = [axes]
    
    for i, N in enumerate(N_range):
        ax = axes[i]
        E = bandsets[N]
        nb = E.shape[1]
        mid = [nb//2-1, nb//2]
        
        for b in range(nb):
            lw = 0.6 if b in mid else 0.3
            color = 'black'
            ax.plot(k_mag, E[:,b], linewidth=lw, color=color)
        
        if i == 0:
            ax.set_ylabel("E (eV)")
        else:
            ax.set_yticks([])
        
        ax.set_title(f"{N} L")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_xlabel("K")
    
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', dpi=300)  # High quality save with tight bbox
        print(f"Plot saved as: {save_as}")  # Confirm save
    else:
        plt.show()

# =============================================================================
# Utility Functions
# =============================================================================

def set_parameters(**kwargs):
    """
    Set global parameters.
    
    Args:
        **kwargs: Parameter name-value pairs (gamma0, gamma1, ..., N_layers, stacking, etc.)
    """
    global gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, E0, Delta
    global N_layers, stacking, K_point, dk, n_k, d_cc
    
    for key, value in kwargs.items():
        if key in globals():
            globals()[key] = value
        else:
            print(f"Warning: Unknown parameter '{key}' ignored.")

def get_info():
    """Print current configuration."""
    print("=== Multilayer Graphene Configuration ===")
    print(f"Stacking: {stacking.upper()}")
    print(f"Layers: {N_layers}")
    print(f"\nTight-binding parameters (eV):")
    print(f"  γ0 = {gamma0:7.3f}  (intralayer)")
    print(f"  γ1 = {gamma1:7.3f}  (vertical dimer)")
    print(f"  γ2 = {gamma2:7.3f}  (next-nearest A-A)")
    print(f"  γ3 = {gamma3:7.3f}  (skew)")
    print(f"  γ4 = {gamma4:7.3f}  (like-sublattice)")
    print(f"  γ5 = {gamma5:7.3f}  (next-nearest B-B)")
    print(f"  E0 = {E0:7.3f}  (on-site shift)")
    print(f"  Δ  = {Delta:7.3f}  (A-B asymmetry)")
    print(f"\nk-path: {n_k} points, dk = {dk}, d_cc = {d_cc} Å")

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Compare ABC vs ABA for 3 layers
    print("Multilayer Graphene Band Structure Calculator")
    print("=" * 50)
    
    # Set to 3-layer ABC
    set_parameters(N_layers=3, stacking='abc')
    get_info()
    
    # Calculate and plot
    print("\nCalculating ABC bands...")
    E_abc, k_mag = calculate_bands()
    plot_bands(E_abc, k_mag, title_suffix="ABC")
    
    # Switch to ABA
    set_parameters(stacking='aba')
    print("\nCalculating ABA bands...")
    E_aba, k_mag = calculate_bands()
    plot_bands(E_aba, k_mag, title_suffix="ABA")
    
    # Panel comparison for ABC
    print("\nGenerating ABC panel comparison...")
    set_parameters(stacking='abc')
    plot_panel_comparison()
