"""
Multilayer graphene band structure calculator using PythTB framework

This module provides the same functionality as multilayer_graphene.py but uses 
the pythtb library for tight-binding calculations. This implementation offers
the benefits of the PythTB framework while maintaining the same API.

Global configuration variables control the calculation parameters:
- gamma0 to gamma5: Tight-binding parameters (eV)
- E0, Delta: On-site energies (eV)
- N_layers: Number of layers
- stacking: 'abc' or 'aba'
- k_path_config: k-point path configuration

Author: Generated from multilayer_graphene.py using PythTB framework
"""

import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model

# =============================================================================
# Global Configuration Variables (same as multilayer_graphene.py)
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
dk = 1.5              # Legacy: k-space range around K (kept for API compatibility)
n_k = 1500            # Number of k-points along default path
d_cc = 1.42           # Carbon-carbon distance (Å)

# =============================================================================
# Helper Functions
# =============================================================================

def _reciprocal_vectors():
    """
    Compute reciprocal lattice vectors (Å⁻¹) for graphene using d_cc.

    Returns:
        tuple: (b1, b2) 2D vectors in Cartesian coordinates (Å⁻¹)
    """
    # Graphene real-space lattice constant a = √3 * d_cc
    a = np.sqrt(3.0) * d_cc
    # Primitive real-space lattice vectors (Å) in standard orientation
    a1 = np.array([a, 0.0])
    a2 = np.array([a/2.0, a*np.sqrt(3.0)/2.0])
    
    # Reciprocal lattice vectors: b_i · a_j = 2π δ_ij
    # For 2D: b1 = 2π (a2_perp) / (a1 × a2), b2 = 2π (a1_perp) / (a1 × a2)
    # where a2_perp = (-a2_y, a2_x) and a1_perp = (-a1_y, a1_x)
    cross_product = a1[0] * a2[1] - a1[1] * a2[0]  # a1 × a2 (z-component)
    
    b1 = 2.0 * np.pi * np.array([-a2[1], a2[0]]) / cross_product
    b2 = 2.0 * np.pi * np.array([a1[1], -a1[0]]) / cross_product
    
    return b1, b2

def get_brillouin_zone(plot=True, ax=None, annotate=True, figsize=(4, 4), save_as=None, show=True):
    """
    Gather first Brillouin zone coordinates and key high-symmetry points, and
    optionally plot the BZ hexagon with Γ, K and M highlighted.

    Args:
        plot: Whether to draw a plot of the BZ
        ax: Optional Matplotlib Axes to draw into (created if None)
        annotate: Whether to add text labels near Γ, K, M
        figsize: Figure size used when creating a new Axes
        save_as: Optional filename to save the figure
        show: Whether to call plt.show() when creating a new figure

    Returns:
        dict: with keys 'b1','b2','Gamma','K_frac','M_frac','K_cart','M_cart','BZ_vertices','fig','ax'
    """
    b1, b2 = _reciprocal_vectors()
    # For graphene, K points are at (±1/3, ±2/3) and (±2/3, ±1/3) combinations that give |K|=4π/(3a)
    K_frac = np.array([
        [ 1/3,  2/3],
        [ 2/3,  1/3], 
        [ 1/3, -1/3],
        [-1/3, -2/3],
        [-2/3, -1/3],
        [-1/3,  1/3],
    ], float)
    # M points are at edge midpoints of first BZ (midpoints between adjacent K points)
    M_frac = np.array([
        [-1/2, -1/2],
        [-1/2,  0.0],
        [ 0.0,  1/2],
        [ 1/2,  1/2],
        [ 1/2,  0.0],
        [ 0.0, -1/2],
    ], float)
    def frac_to_cart(frac):
        return frac[..., 0:1]*b1 + frac[..., 1:2]*b2
    K_cart = frac_to_cart(K_frac).reshape(-1, 2)
    M_cart = frac_to_cart(M_frac).reshape(-1, 2)
    fig = None
    if plot:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        closed_hex = np.vstack([K_cart, K_cart[0]])
        ax.plot(closed_hex[:, 0], closed_hex[:, 1], color='black', linewidth=1.0)
        ax.scatter(K_cart[:, 0], K_cart[:, 1], s=18, c='tab:red', label='K')
        ax.scatter(M_cart[:, 0], M_cart[:, 1], s=18, c='tab:blue', label='M')
        ax.scatter([0.0], [0.0], s=24, c='tab:green', label='Γ', zorder=3)
        if annotate:
            ax.annotate('Γ', (0.0, 0.0), textcoords='offset points', xytext=(6, 4))
            ax.annotate('K', K_cart[0], textcoords='offset points', xytext=(6, 4))
            ax.annotate('M', M_cart[0], textcoords='offset points', xytext=(6, 4))
        ax.set_aspect(1.0)
        ax.set_xlabel('k_x (Å⁻¹)')
        ax.set_ylabel('k_y (Å⁻¹)')
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        if save_as is not None:
            fig_to_save = ax.figure if fig is None else fig
            fig_to_save.savefig(save_as, bbox_inches='tight', dpi=300)
        if fig is not None and show:
            plt.show()

    return {
        'b1': b1, 'b2': b2,
        'Gamma': np.array([0.0, 0.0]),
        'K_frac': K_frac, 'M_frac': M_frac,
        'K_cart': K_cart, 'M_cart': M_cart,
        'BZ_vertices': K_cart,
        'fig': fig if plot else None,
        'ax': ax if plot else None,
    }

def get_k_path():
    """
    Default k-path Γ → K → M with K shifted to x = 0.

    Returns:
        tuple: (k_path_recip, k_mag) with fractional k-points and arc-length axis (Å⁻¹)
    """
    b1, b2 = _reciprocal_vectors()
    Gamma_f = np.array([0.0, 0.0])
    K_f = np.array([1/3, 2/3])  # Use first K point from proper hexagon
    M_f = np.array([1/2, 0.0])
    def f2c(frac):
        return frac[0]*b1 + frac[1]*b2
    Gamma_c, K_c, M_c = f2c(Gamma_f), f2c(K_f), f2c(M_f)
    L1 = np.linalg.norm(K_c - Gamma_c)
    L2 = np.linalg.norm(M_c - K_c)
    n1 = max(2, int(round(n_k * L1 / (L1 + L2))))
    n2 = max(2, n_k - n1)
    seg1 = np.column_stack([
        np.linspace(Gamma_f[0], K_f[0], n1, endpoint=False),
        np.linspace(Gamma_f[1], K_f[1], n1, endpoint=False),
    ])
    seg2 = np.column_stack([
        np.linspace(K_f[0], M_f[0], n2),
        np.linspace(K_f[1], M_f[1], n2),
    ])
    k_path_recip = np.vstack([seg1, seg2])
    k_cart = k_path_recip @ np.vstack([b1, b2])
    deltas = np.vstack([[0.0, 0.0], np.diff(k_cart, axis=0)])
    s = np.cumsum(np.linalg.norm(deltas, axis=1))
    idx_K = seg1.shape[0]
    s_K = s[idx_K]
    k_mag = s - s_K
    return k_path_recip, k_mag

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
# PythTB Model Construction
# =============================================================================

def build_model_abc(N=None):
    """
    Build PythTB model for ABC (rhombohedral) stacked multilayer graphene.
    
    In ABC stacking, all adjacent layer couplings follow the same pattern.
    
    Args:
        N: Number of layers (uses global N_layers if None)
        
    Returns:
        tb_model: PythTB model object
    """
    if N is None:
        N = N_layers
    
    # Create 2D model with 2*N orbitals (A and B for each layer)
    model = tb_model(2, 2, lat=[[1.0, 0.0], [0.0, 1.0]], orb=[[0.0, 0.0] for _ in range(2*N)])
    
    # Set on-site energies
    EA, EB = E0 + Delta, E0
    for n in range(N):
        model.set_onsite(EA, 2*n)      # A sublattice for layer n
        model.set_onsite(EB, 2*n+1)    # B sublattice for layer n
    
    # Intralayer hopping: γ0 with structure factor
    # In PythTB, we need to set hoppings for each k-dependent term
    for n in range(N):
        iA, iB = 2*n, 2*n+1
        # Direct hopping (corresponds to f(k) = 1 term)
        model.set_hop(gamma0, iA, iB, [0, 0])
        # First neighbor (corresponds to exp(i2πkx) term)  
        model.set_hop(gamma0, iA, iB, [1, 0])
        # Second neighbor (corresponds to exp(i2πky) term)
        model.set_hop(gamma0, iA, iB, [0, 1])
    
    # Adjacent layer couplings (ABC: all T_plus)
    for n in range(N-1):
        iA, iB = 2*n, 2*n+1
        jA, jB = 2*(n+1), 2*(n+1)+1
        
        # T_plus matrix elements with structure factor dependencies
        # γ4*f: A_n -> A_{n+1}
        model.set_hop(gamma4, iA, jA, [0, 0])  # Direct term
        model.set_hop(gamma4, iA, jA, [1, 0])  # exp(i2πkx) term
        model.set_hop(gamma4, iA, jA, [0, 1])  # exp(i2πky) term
        
        # γ1: B_n -> A_{n+1} (no structure factor)
        model.set_hop(gamma1, iB, jA, [0, 0])
        
        # γ3*f*: A_n -> B_{n+1} (conjugate structure factor)
        model.set_hop(gamma3, iA, jB, [0, 0])   # Direct term
        model.set_hop(gamma3, iA, jB, [-1, 0])  # exp(-i2πkx) term
        model.set_hop(gamma3, iA, jB, [0, -1])  # exp(-i2πky) term
        
        # γ4*f: B_n -> B_{n+1}
        model.set_hop(gamma4, iB, jB, [0, 0])  # Direct term
        model.set_hop(gamma4, iB, jB, [1, 0])  # exp(i2πkx) term
        model.set_hop(gamma4, iB, jB, [0, 1])  # exp(i2πky) term
    
    # Next-nearest layer couplings (n -> n+2)
    for n in range(N-2):
        iA, iB = 2*n, 2*n+1
        kA, kB = 2*(n+2), 2*(n+2)+1
        # Diagonal coupling: A-A and B-B
        model.set_hop(gamma2, iA, kA, [0, 0])
        model.set_hop(gamma5, iB, kB, [0, 0])
    
    return model

def build_model_aba(N=None):
    """
    Build PythTB model for ABA (Bernal) stacked multilayer graphene.
    
    In ABA stacking, adjacent layer couplings alternate between T_plus and T_minus.
    
    Args:
        N: Number of layers (uses global N_layers if None)
        
    Returns:
        tb_model: PythTB model object
    """
    if N is None:
        N = N_layers
    
    # Create 2D model with 2*N orbitals (A and B for each layer)
    model = tb_model(2, 2, lat=[[1.0, 0.0], [0.0, 1.0]], orb=[[0.0, 0.0] for _ in range(2*N)])
    
    # Set on-site energies
    EA, EB = E0 + Delta, E0
    for n in range(N):
        model.set_onsite(EA, 2*n)      # A sublattice for layer n
        model.set_onsite(EB, 2*n+1)    # B sublattice for layer n
    
    # Intralayer hopping: same as ABC
    for n in range(N):
        iA, iB = 2*n, 2*n+1
        model.set_hop(gamma0, iA, iB, [0, 0])
        model.set_hop(gamma0, iA, iB, [1, 0])
        model.set_hop(gamma0, iA, iB, [0, 1])
    
    # Adjacent layer couplings (ABA: alternating T_plus and T_minus)
    for n in range(N-1):
        iA, iB = 2*n, 2*n+1
        jA, jB = 2*(n+1), 2*(n+1)+1
        
        if n % 2 == 0:
            # T_plus matrix (same as ABC)
            model.set_hop(gamma4, iA, jA, [0, 0])
            model.set_hop(gamma4, iA, jA, [1, 0])
            model.set_hop(gamma4, iA, jA, [0, 1])
            
            model.set_hop(gamma1, iB, jA, [0, 0])
            
            model.set_hop(gamma3, iA, jB, [0, 0])
            model.set_hop(gamma3, iA, jB, [-1, 0])
            model.set_hop(gamma3, iA, jB, [0, -1])
            
            model.set_hop(gamma4, iB, jB, [0, 0])
            model.set_hop(gamma4, iB, jB, [1, 0])
            model.set_hop(gamma4, iB, jB, [0, 1])
        else:
            # T_minus matrix (conjugate structure factors swapped)
            # γ4*f*: A_n -> A_{n+1}
            model.set_hop(gamma4, iA, jA, [0, 0])
            model.set_hop(gamma4, iA, jA, [-1, 0])
            model.set_hop(gamma4, iA, jA, [0, -1])
            
            # γ1: A_n -> B_{n+1}
            model.set_hop(gamma1, iA, jB, [0, 0])
            
            # γ3*f: B_n -> A_{n+1}
            model.set_hop(gamma3, iB, jA, [0, 0])
            model.set_hop(gamma3, iB, jA, [1, 0])
            model.set_hop(gamma3, iB, jA, [0, 1])
            
            # γ4*f*: B_n -> B_{n+1}
            model.set_hop(gamma4, iB, jB, [0, 0])
            model.set_hop(gamma4, iB, jB, [-1, 0])
            model.set_hop(gamma4, iB, jB, [0, -1])
    
    # Next-nearest layer couplings (same as ABC)
    for n in range(N-2):
        iA, iB = 2*n, 2*n+1
        kA, kB = 2*(n+2), 2*(n+2)+1
        model.set_hop(gamma2, iA, kA, [0, 0])
        model.set_hop(gamma5, iB, kB, [0, 0])
    
    return model

def build_model(N=None, stacking_type=None):
    """
    Build PythTB model for multilayer graphene with specified stacking.
    
    Args:
        N: Number of layers (uses global N_layers if None)
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        
    Returns:
        tb_model: PythTB model object
    """
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    
    if stacking_type.lower() == 'abc':
        return build_model_abc(N)
    elif stacking_type.lower() == 'aba':
        return build_model_aba(N)
    else:
        raise ValueError(f"Unknown stacking type: {stacking_type}. Use 'abc' or 'aba'.")

# =============================================================================
# Band Structure Calculation
# =============================================================================

def calculate_bands(N=None, stacking_type=None, k_path=None):
    """
    Calculate band structure for multilayer graphene using PythTB.
    
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
    
    # Build the model
    model = build_model(N, stacking_type)
    
    # Calculate bands along k-path
    # PythTB expects k-points as [k1, k2, ..., kn] where each ki is [kx, ky]
    eigenvals = model.solve_all(k_path_recip)
    
    # Debug: Check the shape and type of eigenvals
    print(f"DEBUG: eigenvals type: {type(eigenvals)}")
    if hasattr(eigenvals, 'shape'):
        print(f"DEBUG: eigenvals shape: {eigenvals.shape}")
        print(f"DEBUG: k_path_recip shape: {k_path_recip.shape}")
        print(f"DEBUG: Expected shape: ({len(k_path_recip)}, {2*N})")
    
    # PythTB solve_all returns (n_bands, n_kpts) but we need (n_kpts, n_bands)
    if hasattr(eigenvals, 'shape'):
        if eigenvals.shape[0] == 2*N and eigenvals.shape[1] == len(k_path_recip):
            # Shape is (n_bands, n_kpts) - transpose needed
            E = np.sort(eigenvals.real.T, axis=1)
        elif eigenvals.shape[0] == len(k_path_recip) and eigenvals.shape[1] == 2*N:
            # Shape is already (n_kpts, n_bands) - just sort
            E = np.sort(eigenvals.real, axis=1)
        else:
            print(f"DEBUG: Unexpected shape {eigenvals.shape}")
            E = np.sort(eigenvals.real, axis=1)
    else:
        # List of arrays - convert to (n_kpts, n_bands)
        E = np.array([np.sort(evals.real) for evals in eigenvals])
    
    print(f"DEBUG: Final E shape: {E.shape}")
    print(f"DEBUG: k_mag shape: {k_mag.shape}")
    
    return E, k_mag

def build_hamiltonian(kx_rec, ky_rec, N=None, stacking_type=None):
    """
    Build Hamiltonian matrix at specific k-point using PythTB.
    
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
    
    # Build the model
    model = build_model(N, stacking_type)
    
    # Get Hamiltonian at specific k-point
    H = model._gen_ham([kx_rec, ky_rec])
    
    return H

# =============================================================================
# Plotting Functions (same interface as multilayer_graphene.py)
# =============================================================================

def plot_bands(E=None, k_mag=None, N=None, stacking_type=None, 
               xlim=None, ylim=(-0.5, 0.5), figsize=(5, 4),
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
    plt.xlabel("k along Γ → K → M")
    plt.title(f"{stacking_type.upper()} | Grüneis TB–GW | {N}-layer (PythTB)")
    if xlim is None:
        # Default: focus around K point (at x=0) along Γ→K→M direction
        k_range = k_mag.max() - k_mag.min()
        plt.xlim(-0.15 * k_range, 0.15 * k_range)  # ±15% of total range around K
    else:
        plt.xlim(xlim)
    plt.ylim(ylim)
    
    # Add high-symmetry point labels on x-axis (K is at x=0)
    current_xlim = plt.gca().get_xlim()
    gamma_pos = k_mag.min()  # Γ at leftmost position
    k_pos = 0.0              # K at center (x=0)
    m_pos = k_mag.max()      # M at rightmost position
    
    # Only show labels for points within current view
    tick_positions = []
    tick_labels = []
    if current_xlim[0] <= gamma_pos <= current_xlim[1]:
        tick_positions.append(gamma_pos)
        tick_labels.append('Γ')
    if current_xlim[0] <= k_pos <= current_xlim[1]:
        tick_positions.append(k_pos)
        tick_labels.append('K')
    if current_xlim[0] <= m_pos <= current_xlim[1]:
        tick_positions.append(m_pos)
        tick_labels.append('M')
    
    if tick_positions:
        plt.xticks(tick_positions, tick_labels)
    
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', dpi=300)  # High quality save with tight bbox
        print(f"Plot saved as: {save_as}")  # Confirm save
    else:
        plt.show()

def plot_panel_comparison(N_range=range(1, 9), stacking_type=None,
                         xlim=None, ylim=(-0.7, 0.7), figsize=(16, 2), save_as=None):
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
        if xlim is None:
            # Default: focus around K point (at x=0) along Γ→K→M direction
            k_range = k_mag.max() - k_mag.min()
            ax.set_xlim(-0.15 * k_range, 0.15 * k_range)  # ±15% of total range around K
        else:
            ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Add high-symmetry point labels on x-axis for bottom panels
        current_xlim = ax.get_xlim()
        gamma_pos = k_mag.min()  # Γ at leftmost position
        k_pos = 0.0              # K at center (x=0)
        m_pos = k_mag.max()      # M at rightmost position
        
        # Only show labels for points within current view
        tick_positions = []
        tick_labels = []
        if current_xlim[0] <= gamma_pos <= current_xlim[1]:
            tick_positions.append(gamma_pos)
            tick_labels.append('Γ')
        if current_xlim[0] <= k_pos <= current_xlim[1]:
            tick_positions.append(k_pos)
            tick_labels.append('K')
        if current_xlim[0] <= m_pos <= current_xlim[1]:
            tick_positions.append(m_pos)
            tick_labels.append('M')
        
        if tick_positions:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        else:
            ax.set_xticks([])
        
        if i == len(N_range) // 2:  # Add xlabel to middle panel
            ax.set_xlabel("k along Γ → K → M")
    
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
    print("=== Multilayer Graphene Configuration (PythTB) ===")
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
    print("Framework: PythTB")

# =============================================================================
# Example Usage and Comparison
# =============================================================================

if __name__ == "__main__":
    # Example: Compare with original implementation
    print("Multilayer Graphene Band Structure Calculator (PythTB)")
    print("=" * 55)
    
    # Set to 3-layer ABC
    set_parameters(N_layers=3, stacking='abc')
    get_info()
    
    # Calculate and plot
    print("\nCalculating ABC bands using PythTB...")
    E_abc, k_mag = calculate_bands()
    plot_bands(E_abc, k_mag)
    
    # Compare with direct matrix method if available
    try:
        import multilayer_graphene as mlg_direct
        print("\nComparing with direct matrix implementation...")
        mlg_direct.set_parameters(N_layers=3, stacking='abc')
        E_direct, k_mag_direct = mlg_direct.calculate_bands()
        
        # Quick comparison at K point
        print(f"PythTB eigenvalues at K: {np.sort(E_abc[n_k//2].real)}")
        print(f"Direct eigenvalues at K: {np.sort(E_direct[n_k//2].real)}")
        print(f"Difference (max): {np.max(np.abs(np.sort(E_abc[n_k//2].real) - np.sort(E_direct[n_k//2].real))):.6f} eV")
        
    except ImportError:
        print("Direct implementation not available for comparison.")
    
    print("\nPythTB implementation completed!")
