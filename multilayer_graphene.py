"""
Multilayer graphene band structure calculator

This module provides functions to calculate band structures for multilayer graphene
with both ABC (rhombohedral) and ABA (Bernal) stacking using a simplified nearest-neighbor model.

Uses only nearest-neighbor tight-binding parameters:
- gamma0: Intralayer nearest-neighbor hopping (eV)
- gamma1: Interlayer vertical dimer hopping (eV)

Global configuration variables control the calculation parameters:
- gamma0, gamma1: Tight-binding parameters (eV)
- E0, Delta: On-site energies (eV)
- N_layers: Number of layers
- stacking: 'abc' or 'aba'
- k_path_config: k-point path configuration

Author: Generated from nnn-bands-ABC.py and nnn-bands-ABA.py, simplified to nearest-neighbor
"""

import csv
import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
try:
    from tqdm import tqdm # type: ignore
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =============================================================================
# Global Configuration Variables
# =============================================================================

# Nearest-neighbor tight-binding parameters [eV]
gamma0 = 3.053   # intralayer nearest-neighbor
gamma1 = 0.403   # interlayer vertical dimer
E0 = 0      # on-site shift
Delta = 0        # A vs B on-site asymmetry

# System configuration
N_layers = 3           # Number of layers
stacking = 'abc'       # 'abc' for rhombohedral or 'aba' for Bernal

# k-path configuration around K point
K_point = np.array([1/3, 1/3], float)  # K point in reciprocal lattice units
dk = 1.5              # Legacy: k-space range around K (kept for API compatibility)
n_k = 3000            # Number of k-points along default path
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

def _frac_to_cart(frac):
    """
    Convert fractional reciprocal coordinates (relative to b1, b2) to Cartesian (Å⁻¹).

    Args:
        frac: (..., 2) array-like fractional coordinates

    Returns:
        ndarray: (..., 2) Cartesian coordinates in Å⁻¹
    """
    b1, b2 = _reciprocal_vectors()
    frac = np.asarray(frac, dtype=float)
    return frac[..., 0:1] * b1 + frac[..., 1:2] * b2

def _real_space_cell_area():
    """
    Area of the 2D primitive unit cell (Å²) used for the honeycomb lattice.
    """
    a = np.sqrt(3.0) * d_cc
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2.0, a * np.sqrt(3.0) / 2.0])
    return abs(a1[0] * a2[1] - a1[1] * a2[0])

def _brillouin_zone_area():
    """
    Area of the 2D first Brillouin zone (Å⁻²) associated with the current lattice.
    """
    b1, b2 = _reciprocal_vectors()
    return abs(np.cross(b1, b2))

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
        dict: with keys:
            - 'b1','b2': reciprocal basis vectors (Å⁻¹)
            - 'Gamma': Γ point (Cartesian Å⁻¹)
            - 'K_frac','M_frac': fractional coords in reciprocal units
            - 'K_cart','M_cart': Cartesian coords (Å⁻¹)
            - 'BZ_vertices': Hexagon vertices (Å⁻¹) ordered around Γ
            - 'fig','ax': Matplotlib Figure and Axes (if plot=True)
    """
    b1, b2 = _reciprocal_vectors()

    # Fractional coordinates (relative to b1, b2) of K and M families
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
    BZ_vertices = K_cart  # Hexagon corners are the K points

    fig = None
    if plot:
        # Create axes if not provided
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        
        # Draw BZ hexagon (close the loop)
        closed_hex = np.vstack([BZ_vertices, BZ_vertices[0]])
        ax.plot(closed_hex[:, 0], closed_hex[:, 1], color='black', linewidth=1.0)
        
        # Plot all K and M points and mark Γ
        ax.scatter(K_cart[:, 0], K_cart[:, 1], s=18, c='tab:red', label='K')
        ax.scatter(M_cart[:, 0], M_cart[:, 1], s=18, c='tab:blue', label='M')
        ax.scatter([0.0], [0.0], s=24, c='tab:green', label='Γ', zorder=3)
        
        # Optional annotations (label a representative K and M)
        if annotate:
            ax.annotate('Γ', (0.0, 0.0), textcoords='offset points', xytext=(6, 4))
            ax.annotate('K', K_cart[0], textcoords='offset points', xytext=(6, 4))
            ax.annotate('M', M_cart[0], textcoords='offset points', xytext=(6, 4))
        
        # Styling for clarity
        ax.set_aspect(1.0)
        ax.set_xlabel('k_x (Å⁻¹)')
        ax.set_ylabel('k_y (Å⁻¹)')
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        
        # Save/show if we created the figure here
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
        'BZ_vertices': BZ_vertices,
        'fig': fig if plot else None,
        'ax': ax if plot else None,
    }

def get_k_path(path_type='gkm'):
    """
    Generate k-path with K shifted to x = 0.

    Args:
        path_type: 'gkm' for Γ → K → M (default) or 'gkg' for Γ → K → Γ

    Returns:
        tuple: (k_path_recip, k_mag)
            - k_path_recip: path in reciprocal lattice units (fractional b1,b2)
            - k_mag: 1D coordinate (Å⁻¹) along the path with K at 0
    """
    b1, b2 = _reciprocal_vectors()

    # High-symmetry fractional coordinates
    Gamma_f = np.array([0.0, 0.0])
    K_f = np.array([1/3, 2/3])  # Use first K point from proper hexagon
    M_f = np.array([1/2, 0.0])

    # Convert to Cartesian (Å⁻¹) for arc-length allocation and x-axis
    def f2c(frac):
        return frac[0]*b1 + frac[1]*b2

    Gamma_c, K_c, M_c = f2c(Gamma_f), f2c(K_f), f2c(M_f)

    if path_type.lower() == 'gkm':
        # Γ → K → M path
        # Allocate points proportionally to segment lengths (keep total n_k)
        L1 = np.linalg.norm(K_c - Gamma_c)
        L2 = np.linalg.norm(M_c - K_c)
        n1 = max(2, int(round(n_k * L1 / (L1 + L2))))
        n2 = max(2, n_k - n1)

        # Build path in fractional coordinates; include K once at segment junction
        seg1 = np.column_stack([
            np.linspace(Gamma_f[0], K_f[0], n1, endpoint=False),
            np.linspace(Gamma_f[1], K_f[1], n1, endpoint=False),
        ])
        seg2 = np.column_stack([
            np.linspace(K_f[0], M_f[0], n2),
            np.linspace(K_f[1], M_f[1], n2),
        ])
        k_path_recip = np.vstack([seg1, seg2])

        # Cartesian coordinates along the polyline (Å⁻¹)
        k_cart = k_path_recip @ np.vstack([b1, b2])

        # Cumulative arc-length along Γ→K→M
        deltas = np.vstack([[0.0, 0.0], np.diff(k_cart, axis=0)])
        s = np.cumsum(np.linalg.norm(deltas, axis=1))

        # Index of K is at the junction (end of seg1)
        idx_K = seg1.shape[0]
        s_K = s[idx_K]

        # Shift so that K maps to x = 0; Γ negative, M positive
        k_mag = s - s_K

    elif path_type.lower() == 'gkg':
        # Γ → K → Γ path
        # Allocate points equally between the two segments
        n1 = max(2, n_k // 2)
        n2 = max(2, n_k - n1)

        # Build path in fractional coordinates
        seg1 = np.column_stack([
            np.linspace(Gamma_f[0], K_f[0], n1, endpoint=False),
            np.linspace(Gamma_f[1], K_f[1], n1, endpoint=False),
        ])
        seg2 = np.column_stack([
            np.linspace(K_f[0], Gamma_f[0], n2),
            np.linspace(K_f[1], Gamma_f[1], n2),
        ])
        k_path_recip = np.vstack([seg1, seg2])

        # Cartesian coordinates along the polyline (Å⁻¹)
        k_cart = k_path_recip @ np.vstack([b1, b2])

        # Cumulative arc-length along Γ→K→Γ
        deltas = np.vstack([[0.0, 0.0], np.diff(k_cart, axis=0)])
        s = np.cumsum(np.linalg.norm(deltas, axis=1))

        # Index of K is at the junction (end of seg1)
        idx_K = seg1.shape[0]
        s_K = s[idx_K]

        # Shift so that K maps to x = 0; Γ points negative and positive
        k_mag = s - s_K

    else:
        raise ValueError(f"Unknown path_type: {path_type}. Use 'gkm' or 'gkg'.")

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
        dict: Dictionary containing gamma parameters and on-site energies
    """
    return {
        'gamma0': gamma0, 'gamma1': gamma1,
        'E0': E0, 'Delta': Delta
    }

# =============================================================================
# Hamiltonian Construction
# =============================================================================

def build_hamiltonian_abc(kx_rec, ky_rec, N=None):
    """
    Build Hamiltonian for ABC (rhombohedral) stacked multilayer graphene.
    
    Uses only nearest-neighbor intralayer (gamma0) and interlayer vertical dimer (gamma1) coupling.
    In ABC stacking, only the gamma1 coupling is used between adjacent layers.
    
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
    
    # Adjacent layers: n -> n+1, only gamma1 coupling (B_n <-> A_{n+1})
    for n in range(N-1):
        iA, iB = 2*n, 2*n+1
        jA, jB = 2*(n+1), 2*(n+1)+1
        
        # Only vertical dimer coupling: B_n <-> A_{n+1}
        H[iB, jA] += gamma1
        H[jA, iB] += gamma1
    
    return H

def build_hamiltonian_aba(kx_rec, ky_rec, N=None):
    """
    Build Hamiltonian for ABA (Bernal) stacked multilayer graphene.
    
    Uses only nearest-neighbor intralayer (gamma0) and interlayer vertical dimer (gamma1) coupling.
    In ABA stacking, the gamma1 coupling alternates between different sublattice pairs.
    
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
    
    # Adjacent layers: n -> n+1, alternating gamma1 coupling
    for n in range(N-1):
        iA, iB = 2*n, 2*n+1
        jA, jB = 2*(n+1), 2*(n+1)+1
        
        if n % 2 == 0:
            # Even layers: B_n <-> A_{n+1}
            H[iB, jA] += gamma1
            H[jA, iB] += gamma1
        else:
            # Odd layers: A_n <-> B_{n+1}
            H[iA, jB] += gamma1
            H[jB, iA] += gamma1
    
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

def calculate_bands(N=None, stacking_type=None, k_path=None, path_type='gkm'):
    """
    Calculate band structure for multilayer graphene.
    
    Args:
        N: Number of layers (uses global N_layers if None)
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        k_path: (k_path_recip, k_mag) tuple (generates from globals if None)
        path_type: 'gkm' for Γ → K → M (default) or 'gkg' for Γ → K → Γ
        
    Returns:
        tuple: (eigenvalues, k_mag) where eigenvalues has shape (n_k, 2N)
               and k_mag is for plotting
    """
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    if k_path is None:
        k_path_recip, k_mag = get_k_path(path_type)
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
               path_type='gkm', xlim=None, ylim=(-0.5, 0.5), figsize=(5, 4),
               highlight_middle=True, save_as=None):
    """
    Plot band structure for single layer number.
    
    Args:
        E: Eigenvalues array (calculates if None)
        k_mag: k-magnitude array for x-axis (calculates if None)
        N: Number of layers (uses global N_layers if None)
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        path_type: 'gkm' for Γ → K → M (default) or 'gkg' for Γ → K → Γ
        xlim, ylim: Plot limits
        figsize: Figure size
        highlight_middle: Whether to highlight middle bands with thicker lines
        save_as: Filename to save plot (e.g., 'bands.svg', 'bands.png'). If None, just show.
    """
    if E is None or k_mag is None:
        E, k_mag = calculate_bands(N, stacking_type, k_path=get_k_path(path_type))
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    
    plt.figure(figsize=figsize)
    nb = E.shape[1]
    mid = [nb//2-1, nb//2] if highlight_middle else []
    
    for b in range(nb):
        lw = 0.8 if b in mid else 0.5
        col = 'red' if b in mid else 'black'
        plt.plot(k_mag, E[:,b], linewidth=lw, color=col)  # Plot all bands in black
    
    plt.ylabel("Energy (eV)")
    
    # Set x-axis label based on path type
    if path_type.lower() == 'gkm':
        plt.xlabel("k along Γ → K → M")
    elif path_type.lower() == 'gkg':
        plt.xlabel("k along Γ → K → Γ")
    else:
        plt.xlabel("k along path")
    
    plt.title(f"{stacking_type.upper()} | {N}-layer")
    if xlim is None:
        # Default: focus around K point (at x=0) along the path direction
        k_range = k_mag.max() - k_mag.min()
        plt.xlim(-0.15 * k_range, 0.15 * k_range)  # ±15% of total range around K
    else:
        plt.xlim(xlim)
    plt.ylim(ylim)
    
    # Add high-symmetry point labels on x-axis (K is at x=0)
    current_xlim = plt.gca().get_xlim()
    gamma_pos = k_mag.min()  # Γ at leftmost position
    k_pos = 0.0              # K at center (x=0)
    
    if path_type.lower() == 'gkm':
        m_pos = k_mag.max()  # M at rightmost position
    else:  # gkg
        m_pos = k_mag.max()  # Second Γ at rightmost position
    
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
        if path_type.lower() == 'gkm':
            tick_labels.append('M')
        else:  # gkg
            tick_labels.append('Γ')
    
    if tick_positions:
        plt.xticks(tick_positions, tick_labels)
    
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', dpi=300)  # High quality save with tight bbox
        print(f"Plot saved as: {save_as}")  # Confirm save
    else:
        plt.show()

def plot_panel_comparison(N_range=range(1, 9), stacking_type=None, path_type='gkm',
                         xlim=None, ylim=(-0.7, 0.7), figsize=(16, 2), 
                         highlight_middle=True, save_as=None):
    """
    Plot comparison panels for different layer numbers.
    
    Args:
        N_range: Range of layer numbers to plot
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        path_type: 'gkm' for Γ → K → M (default) or 'gkg' for Γ → K → Γ
        xlim, ylim: Plot limits
        figsize: Figure size
        highlight_middle: Whether to highlight middle bands with thicker lines
        save_as: Filename to save plot (e.g., 'comparison.svg', 'comparison.png'). If None, just show.
    """
    if stacking_type is None:
        stacking_type = stacking
    
    # Calculate all band structures
    k_path_recip, k_mag = get_k_path(path_type)
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
        mid = [nb//2-1, nb//2] if highlight_middle else []
        
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
        
        if path_type.lower() == 'gkm':
            m_pos = k_mag.max()  # M at rightmost position
        else:  # gkg
            m_pos = k_mag.max()  # Second Γ at rightmost position
        
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
            if path_type.lower() == 'gkm':
                tick_labels.append('M')
            else:  # gkg
                tick_labels.append('Γ')
        
        if tick_positions:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        else:
            ax.set_xticks([])
        
        if i == len(N_range) // 2:  # Add xlabel to middle panel
            if path_type.lower() == 'gkm':
                ax.set_xlabel("k along Γ → K → M")
            elif path_type.lower() == 'gkg':
                ax.set_xlabel("k along Γ → K → Γ")
            else:
                ax.set_xlabel("k along path")
    
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', dpi=300)  # High quality save with tight bbox
        print(f"Plot saved as: {save_as}")  # Confirm save
    else:
        plt.show()

def plot_band_density(band_index, N=None, stacking_type=None, 
                      center_frac=(1/3, 2/3), half_width_frac=0.12, resolution=0.0008,
                      fast=False, xlim=None, ylim=None,
                      cmap='viridis', shading='auto', vmin=None, vmax=None,
                      figsize=(5, 4), save_as=None):
    """
    Plot a 2D density map E_b(k_x, k_y) of a specific band around a center point.
    
    Args:
        band_index: Integer index of the band to plot (0-based; wraps modulo total bands)
        N: Number of layers (uses global N_layers if None)
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        center_frac: Center of grid in fractional reciprocal units (default near K: (1/3, 2/3))
        half_width_frac: Half-width of the grid box in fractional units (controls k-space coverage)
        resolution: Spacing between k-points in fractional units (controls grid density)
        fast: If True, use coarser resolution for faster computation
        xlim, ylim: Limits in Cartesian Å⁻¹ (tuples). If None, set from grid extents.
        cmap, shading, vmin, vmax: Matplotlib pcolormesh parameters
        figsize: Figure size
        save_as: Optional path to save the figure
    """
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    
    center_frac = np.asarray(center_frac, dtype=float)
    
    # Adjust parameters for fast mode
    if fast:
        resolution = resolution * 4  # Coarser resolution for speed
        print(f"Fast mode: using coarser resolution = {resolution:.4f}")
    
    # Determine the actual calculation region
    if xlim is not None or ylim is not None:
        # Convert xlim/ylim (Cartesian Å⁻¹) to fractional coordinates to determine calculation bounds
        b1, b2 = _reciprocal_vectors()
        
        if xlim is not None and ylim is not None:
            # Convert corner points to fractional coordinates
            corners_cart = np.array([
                [xlim[0], ylim[0]], [xlim[1], ylim[0]], 
                [xlim[0], ylim[1]], [xlim[1], ylim[1]]
            ])
            # Transform to fractional: solve [b1, b2] @ frac = cart
            B_matrix = np.vstack([b1, b2]).T
            corners_frac = np.linalg.solve(B_matrix, corners_cart.T).T
            
            # Find bounding box in fractional coordinates
            kx_min, kx_max = corners_frac[:, 0].min(), corners_frac[:, 0].max()
            ky_min, ky_max = corners_frac[:, 1].min(), corners_frac[:, 1].max()
            
            # Add small buffer to ensure we cover the display region
            buffer_frac = resolution * 2
            kx_min -= buffer_frac
            kx_max += buffer_frac
            ky_min -= buffer_frac
            ky_max += buffer_frac
            
            print(f"Limiting calculation to display region: xlim={xlim}, ylim={ylim}")
        else:
            # Only one limit specified - use default coverage for the other
            kx_min = center_frac[0] - half_width_frac
            kx_max = center_frac[0] + half_width_frac
            ky_min = center_frac[1] - half_width_frac
            ky_max = center_frac[1] + half_width_frac
            print(f"Using partial display limits with default coverage")
    else:
        # No display limits - use full coverage
        kx_min = center_frac[0] - half_width_frac
        kx_max = center_frac[0] + half_width_frac
        ky_min = center_frac[1] - half_width_frac
        ky_max = center_frac[1] + half_width_frac
    
    # Calculate grid size from resolution and actual calculation bounds
    n_kx = max(2, int(np.ceil((kx_max - kx_min) / resolution)) + 1)
    n_ky = max(2, int(np.ceil((ky_max - ky_min) / resolution)) + 1)
    
    # Report what we're computing
    coverage_x = kx_max - kx_min
    coverage_y = ky_max - ky_min
    total_points = n_kx * n_ky
    print(f"Computing {n_kx}×{n_ky} = {total_points:,} k-points")
    print(f"Coverage: {coverage_x:.3f}×{coverage_y:.3f} fractional units, resolution = {resolution:.4f}")
    
    # Create k-grid in fractional coordinates
    kx_lin = np.linspace(kx_min, kx_max, n_kx)
    ky_lin = np.linspace(ky_min, ky_max, n_ky)
    kx_rec, ky_rec = np.meshgrid(kx_lin, ky_lin, indexing='xy')
    
    # Convert to Cartesian for plotting
    k_cart = _frac_to_cart(np.dstack([kx_rec, ky_rec]))
    kx_cart, ky_cart = k_cart[..., 0], k_cart[..., 1]
    
    # Compute energies on the grid
    E = np.empty((n_ky, n_kx, 2 * N), dtype=float)
    total_points = n_kx * n_ky
    
    # Use progress bar for large computations
    if HAS_TQDM and total_points > 5000:
        # Use tqdm progress bar
        with tqdm(total=n_ky, desc="Computing bands", unit="row") as pbar:
            for iy in range(n_ky):
                for ix in range(n_kx):
                    H = build_hamiltonian(kx_rec[iy, ix], ky_rec[iy, ix], N=N, stacking_type=stacking_type)
                    evals = np.linalg.eigvalsh(H)
                    E[iy, ix, :] = np.sort(evals.real)
                pbar.update(1)
    else:
        # Simple computation without progress bar
        if total_points > 20000:
            print(f"Computing band structure on {n_kx}×{n_ky} = {total_points:,} k-points...")
        
        for iy in range(n_ky):
            for ix in range(n_kx):
                H = build_hamiltonian(kx_rec[iy, ix], ky_rec[iy, ix], N=N, stacking_type=stacking_type)
                evals = np.linalg.eigvalsh(H)
                E[iy, ix, :] = np.sort(evals.real)
    
    # Select the band
    b = int(band_index) % (2 * N)
    Z = E[..., b]
    
    # Determine colormap limits from the actual data
    if vmin is None or vmax is None:
        if xlim is not None and ylim is not None:
            # Find data points within the display limits
            mask = ((kx_cart >= xlim[0]) & (kx_cart <= xlim[1]) & 
                   (ky_cart >= ylim[0]) & (ky_cart <= ylim[1]))
            if np.any(mask):
                Z_visible = Z[mask]
                auto_vmin = np.min(Z_visible) if vmin is None else vmin
                auto_vmax = np.max(Z_visible) if vmax is None else vmax
            else:
                # Fallback if no points in display region
                auto_vmin = np.min(Z) if vmin is None else vmin
                auto_vmax = np.max(Z) if vmax is None else vmax
        else:
            # Use full data range
            auto_vmin = np.min(Z) if vmin is None else vmin
            auto_vmax = np.max(Z) if vmax is None else vmax
        
        print(f"Auto colormap range: [{auto_vmin:.4f}, {auto_vmax:.4f}] eV")
    else:
        auto_vmin, auto_vmax = vmin, vmax
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(kx_cart, ky_cart, Z, cmap=cmap, shading=shading, vmin=auto_vmin, vmax=auto_vmax)
    ax.set_xlabel('k_x (Å⁻¹)')
    ax.set_ylabel('k_y (Å⁻¹)')
    ax.set_aspect('equal')
    ax.set_title(f"{stacking_type.upper()} | {N}-layer | band {b}")
    
    # Set axis limits to match calculated region
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Energy (eV)')
    
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', dpi=300)
        print(f"Plot saved as: {save_as}")
    else:
        plt.show()

# =============================================================================
# Density of States
# =============================================================================

def _sample_bz_spectrum(N, stacking_type, grid_shape, use_tqdm=None):
    """
    Sample the Brillouin zone on a uniform fractional grid and return eigenvalues.

    Args:
        N: Number of layers
        stacking_type: 'abc' or 'aba'
        grid_shape: Tuple (n_kx, n_ky) for the sampling grid
        use_tqdm: Force enabling/disabling tqdm progress (None = auto)

    Returns:
        dict: Contains sampled eigenvalues and bookkeeping information:
              'eigenvalues': array with shape (n_kx * n_ky, 2N)
              'total_kpoints': n_kx * n_ky
              'grid_shape': (n_kx, n_ky)
    """
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must be a tuple of two integers (n_kx, n_ky).")
    n_kx, n_ky = int(grid_shape[0]), int(grid_shape[1])
    if n_kx <= 0 or n_ky <= 0:
        raise ValueError("grid_shape entries must be positive integers.")

    total_kpoints = n_kx * n_ky
    kx_vals = np.linspace(0.0, 1.0, n_kx, endpoint=False)
    ky_vals = np.linspace(0.0, 1.0, n_ky, endpoint=False)

    eigenvalues = np.empty((total_kpoints, 2 * N), dtype=float)

    enable_tqdm = HAS_TQDM if use_tqdm is None else bool(use_tqdm)
    show_progress = enable_tqdm and total_kpoints >= 5000
    iterator = enumerate(ky_vals)
    if show_progress:
        iterator = enumerate(tqdm(ky_vals, desc="Sampling BZ", unit="ky"))

    idx = 0
    for iy, ky in iterator:
        for kx in kx_vals:
            H = build_hamiltonian(kx, ky, N=N, stacking_type=stacking_type)
            evals = np.linalg.eigvalsh(H).real
            eigenvalues[idx] = np.sort(evals)
            idx += 1

    return {
        'eigenvalues': eigenvalues,
        'total_kpoints': total_kpoints,
        'grid_shape': (n_kx, n_ky),
    }

def calculate_density_of_states(energy_range=None, N=None, stacking_type=None,
                                grid_shape=(200, 200), num_energy_points=400,
                                gaussian_sigma=0.01, use_tqdm=None):
    """
    Calculate the density of states over an energy range.

    Args:
        energy_range: Tuple (emin, emax) in eV. If None, inferred from spectrum.
        N: Number of layers (defaults to global N_layers)
        stacking_type: 'abc' or 'aba' (defaults to global stacking)
        grid_shape: Tuple (n_kx, n_ky) sampling the Brillouin zone
        num_energy_points: Number of energy samples / histogram bins
        gaussian_sigma: Optional Gaussian broadening (eV). Use 0 or None to disable.
        use_tqdm: Force enabling/disabling tqdm progress (None = auto)

    Returns:
        dict: DOS data with keys:
              'energy', 'dos_per_unit_cell', 'dos_per_area', 'bin_width',
              'counts', 'total_kpoints', 'grid_shape', 'parameters'
    """
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking
    if num_energy_points < 2:
        raise ValueError("num_energy_points must be at least 2.")

    sample = _sample_bz_spectrum(N, stacking_type, grid_shape, use_tqdm=use_tqdm)
    eigenvalues = sample['eigenvalues'].ravel()
    total_kpoints = sample['total_kpoints']

    if energy_range is None:
        emin, emax = float(np.min(eigenvalues)), float(np.max(eigenvalues))
        span = max(emax - emin, 1e-6)
        padding = 0.05 * span
        e_min = emin - padding
        e_max = emax + padding
    else:
        e_min, e_max = map(float, energy_range)
        if e_max < e_min:
            e_min, e_max = e_max, e_min

    energy_edges = np.linspace(e_min, e_max, num_energy_points + 1)
    counts, _ = np.histogram(eigenvalues, bins=energy_edges)
    captured = counts.sum()
    total_states = eigenvalues.size
    if captured < total_states:
        missing = total_states - captured
        print(f"Warning: {missing} states lie outside the specified energy_range.")

    bin_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    bin_width = energy_edges[1] - energy_edges[0]
    states_per_unit_cell = counts / total_kpoints
    dos_per_unit_cell = states_per_unit_cell / bin_width

    real_space_area = _real_space_cell_area()
    states_per_area = states_per_unit_cell / real_space_area
    dos_per_area = states_per_area / bin_width

    if gaussian_sigma is not None and gaussian_sigma > 0.0:
        sigma_bins = gaussian_sigma / bin_width
        if sigma_bins > 0.0:
            radius = max(1, int(np.ceil(4.0 * sigma_bins)))
            offsets = np.arange(-radius, radius + 1, dtype=float)
            kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
            kernel /= kernel.sum()
            dos_per_unit_cell = np.convolve(dos_per_unit_cell, kernel, mode='same')
            dos_per_area = np.convolve(dos_per_area, kernel, mode='same')

    return {
        'energy': bin_centers,
        'dos_per_unit_cell': dos_per_unit_cell,
        'dos_per_area': dos_per_area,
        'bin_width': bin_width,
        'counts': counts,
        'total_kpoints': total_kpoints,
        'grid_shape': sample['grid_shape'],
        'parameters': {
            'N_layers': N,
            'stacking': stacking_type,
            'energy_range': (e_min, e_max),
            'gaussian_sigma': gaussian_sigma,
            'grid_shape': sample['grid_shape'],
            'num_energy_points': num_energy_points,
        }
    }

def calculate_density_of_states_window(energy_min, energy_max, N=None, stacking_type=None,
                                       grid_shape=(200, 200), use_tqdm=None):
    """
    Approximate the density of states averaged over a given energy window.

    The calculation samples the primitive Brillouin zone with a uniform grid in
    fractional reciprocal coordinates and counts eigenvalues that fall inside the
    requested energy window. The result is returned both per-layer unit cell and
    per absolute area using the in-plane lattice vectors defined for the model.

    Args:
        energy_min: Lower bound of the energy window (eV)
        energy_max: Upper bound of the energy window (eV)
        N: Number of layers (defaults to global N_layers)
        stacking_type: 'abc' or 'aba' (defaults to global stacking)
        grid_shape: Tuple (n_kx, n_ky) controlling the sampling grid in fractional k-space
        use_tqdm: Force enabling/disabling tqdm progress bar (None = auto)

    Returns:
        dict: Contains averaged DOS and bookkeeping fields with keys:
              'average_dos_per_unit_cell' (states per unit cell per eV),
              'average_dos_per_area' (states per Å² per eV),
              'integrated_states_per_unit_cell',
              'integrated_states_per_area',
              'raw_state_count', 'grid_shape', 'energy_window',
              'window_width', 'N_layers', 'stacking', 'bz_area', 'real_space_cell_area'
    """
    if N is None:
        N = N_layers
    if stacking_type is None:
        stacking_type = stacking

    e_min = float(energy_min)
    e_max = float(energy_max)
    if e_max < e_min:
        e_min, e_max = e_max, e_min

    window_width = e_max - e_min
    if window_width <= 0.0:
        raise ValueError("energy_max must be greater than energy_min for DOS calculation.")

    sample = _sample_bz_spectrum(N, stacking_type, grid_shape, use_tqdm=use_tqdm)
    eigenvalues = sample['eigenvalues'].ravel()
    total_kpoints = sample['total_kpoints']

    raw_count = int(np.count_nonzero((eigenvalues >= e_min) & (eigenvalues <= e_max)))

    # Each k-point represents the same fraction of the Brillouin zone
    states_per_unit_cell = raw_count / total_kpoints

    # Convert to DOS per real-space area using lattice vectors consistent with _reciprocal_vectors()
    real_space_area = _real_space_cell_area()
    states_per_area = states_per_unit_cell / real_space_area

    avg_dos_unit_cell = states_per_unit_cell / window_width
    avg_dos_area = states_per_area / window_width

    bz_area = _brillouin_zone_area()

    return {
        'average_dos_per_unit_cell': avg_dos_unit_cell,
        'average_dos_per_area': avg_dos_area,
        'integrated_states_per_unit_cell': states_per_unit_cell,
        'integrated_states_per_area': states_per_area,
        'raw_state_count': raw_count,
        'grid_shape': sample['grid_shape'],
        'energy_window': (e_min, e_max),
        'window_width': window_width,
        'N_layers': N,
        'stacking': stacking_type,
        'bz_area': bz_area,
        'real_space_cell_area': real_space_area,
    }

def _save_dos_to_csv(csv_path, dos_data, per_area=False, metadata_extra=None):
    """
    Save DOS arrays and metadata to a CSV file with a JSON metadata header.
    """
    metadata = dict(dos_data.get('parameters', {}))
    metadata.update({
        'bin_width': dos_data.get('bin_width'),
        'total_kpoints': dos_data.get('total_kpoints'),
        'grid_shape': tuple(dos_data.get('grid_shape', ())),
        'per_area': bool(per_area),
    })
    if metadata_extra:
        metadata.update(metadata_extra)
    metadata['plot_kwargs'] = {
        'energy_range': metadata.get('energy_range'),
        'N': metadata.get('N_layers'),
        'stacking_type': metadata.get('stacking'),
        'grid_shape': metadata.get('grid_shape'),
        'num_energy_points': metadata.get('num_energy_points'),
        'gaussian_sigma': metadata.get('gaussian_sigma'),
        'per_area': metadata.get('per_area'),
    }
    metadata['columns'] = [
        'energy_eV',
        'dos_per_unit_cell_states_per_eV',
        'dos_per_area_states_per_eV_A2',
    ]

    energy = np.asarray(dos_data['energy'])
    dos_uc = np.asarray(dos_data['dos_per_unit_cell'])
    dos_area = np.asarray(dos_data['dos_per_area'])

    with open(csv_path, 'w', newline='') as fh:
        fh.write(f"# metadata: {json.dumps(metadata, sort_keys=True)}\n")
        writer = csv.writer(fh)
        writer.writerow(metadata['columns'])
        for row in zip(energy, dos_uc, dos_area):
            writer.writerow(row)

def load_dos_from_csv(csv_path):
    """
    Load DOS data and metadata saved by plot_dos() from a CSV file.

    Args:
        csv_path: Path to the CSV file produced by plot_dos(csv_path=...).

    Returns:
        dict: {
            'energy': np.ndarray,
            'dos_per_unit_cell': np.ndarray,
            'dos_per_area': np.ndarray,
            'metadata': dict (includes 'plot_kwargs' for plot_dos)
        }
    """
    with open(csv_path, 'r', newline='') as fh:
        first_line = fh.readline().strip()
        if not first_line.startswith("# metadata:"):
            raise ValueError("CSV file does not contain metadata header.")
        metadata_json = first_line[len("# metadata:"):].strip()
        metadata = json.loads(metadata_json)
        if 'grid_shape' in metadata:
            metadata['grid_shape'] = tuple(metadata['grid_shape'])
        if 'energy_range' in metadata:
            metadata['energy_range'] = tuple(metadata['energy_range'])
        if 'plot_kwargs' in metadata and metadata['plot_kwargs'] is not None:
            pk = metadata['plot_kwargs']
            if 'grid_shape' in pk and pk['grid_shape'] is not None:
                pk['grid_shape'] = tuple(pk['grid_shape'])
            if 'energy_range' in pk and pk['energy_range'] is not None:
                pk['energy_range'] = tuple(pk['energy_range'])

        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV file is missing data header row.")
        expected_cols = metadata.get('columns')
        if expected_cols and list(header) != list(expected_cols):
            raise ValueError("CSV columns do not match metadata description.")

        energy = []
        dos_uc = []
        dos_area = []
        for row in reader:
            if not row:
                continue
            energy.append(float(row[0]))
            dos_uc.append(float(row[1]))
            dos_area.append(float(row[2]))

    energy_arr = np.asarray(energy, dtype=float)
    dos_uc_arr = np.asarray(dos_uc, dtype=float)
    dos_area_arr = np.asarray(dos_area, dtype=float)

    return {
        'energy': energy_arr,
        'dos_per_unit_cell': dos_uc_arr,
        'dos_per_area': dos_area_arr,
        'metadata': metadata,
    }

def plot_dos(energy_range=None, N=None, stacking_type=None,
             grid_shape=(200, 200), num_energy_points=400,
             gaussian_sigma=0.01, use_tqdm=None, per_area=False,
             ax=None, figsize=(5, 4), label=None, show=True,
             return_data=False, csv_path=None, metadata_extra=None):
    """
    Plot the density of states for the multilayer graphene model.

    Args:
        energy_range: Tuple (emin, emax) for plotting range (eV). Inferred if None.
        N: Number of layers (defaults to global N_layers)
        stacking_type: 'abc' or 'aba' (defaults to global stacking)
        grid_shape: Tuple (n_kx, n_ky) sampling the Brillouin zone
        num_energy_points: Number of histogram bins / plotted samples
        gaussian_sigma: Optional Gaussian smoothing width (eV)
        use_tqdm: Force enabling/disabling tqdm progress (None = auto)
        per_area: If True, plot DOS in states/(eV Å²). Otherwise per unit cell.
        ax: Optional Matplotlib Axes to draw on
        figsize: Figure size when creating new axes
        label: Optional legend label
        show: Call plt.show() when creating a new figure
        return_data: If True, return the DOS dictionary used for plotting
        csv_path: Optional path. When provided, save the DOS data to a CSV file.
        metadata_extra: Optional dict merged into the saved metadata header.

    Returns:
        dict or None: DOS dictionary when return_data is True, else None.
    """
    dos_data = calculate_density_of_states(
        energy_range=energy_range,
        N=N,
        stacking_type=stacking_type,
        grid_shape=grid_shape,
        num_energy_points=num_energy_points,
        gaussian_sigma=gaussian_sigma,
        use_tqdm=use_tqdm,
    )

    y_values = dos_data['dos_per_area'] if per_area else dos_data['dos_per_unit_cell']
    units = 'states/(eV Å²)' if per_area else 'states/(eV unit cell)'

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(dos_data['energy'], y_values, label=label)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(f"DOS ({units})")

    N_plot = dos_data['parameters']['N_layers']
    stacking_plot = dos_data['parameters']['stacking'].upper()
    ax.set_title(f"{stacking_plot} | {N_plot}-layer DOS")

    if label is not None:
        ax.legend()

    if created_fig:
        fig.tight_layout()
        if show:
            plt.show()

    if csv_path is not None:
        metadata_payload = dict(metadata_extra) if metadata_extra else {}
        _save_dos_to_csv(csv_path, dos_data, per_area=per_area, metadata_extra=metadata_payload)

    if return_data:
        return dos_data

    return None

# =============================================================================
# Utility Functions
# =============================================================================

def set_parameters(**kwargs):
    """
    Set global parameters.
    
    Args:
        **kwargs: Parameter name-value pairs (gamma0, gamma1, N_layers, stacking, etc.)
    """
    global gamma0, gamma1, E0, Delta
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
    print(f"  γ0 = {gamma0:7.3f}  (intralayer nearest-neighbor)")
    print(f"  γ1 = {gamma1:7.3f}  (interlayer vertical dimer)")
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
