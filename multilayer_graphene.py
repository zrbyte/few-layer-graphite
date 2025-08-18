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
                         xlim=None, ylim=(-0.7, 0.7), figsize=(16, 2), save_as=None):
    """
    Plot comparison panels for different layer numbers.
    
    Args:
        N_range: Range of layer numbers to plot
        stacking_type: 'abc' or 'aba' (uses global stacking if None)
        path_type: 'gkm' for Γ → K → M (default) or 'gkg' for Γ → K → Γ
        xlim, ylim: Plot limits
        figsize: Figure size
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
