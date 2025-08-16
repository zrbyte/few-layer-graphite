#!/usr/bin/env python3
"""
Example usage of the multilayer_graphene module.

This script demonstrates how to use the unified multilayer graphene
band structure calculator with global configuration variables.

Note: For PythTB-based implementation, see multilayer_graphene_pythtb.py
and compare_implementations.py for detailed comparison between approaches.
"""

import multilayer_graphene as mlg
import matplotlib.pyplot as plt

def main():
    """Main example function."""
    print("Multilayer Graphene Band Structure Example")
    print("=" * 45)
    
    # Show current configuration
    mlg.get_info()
    
    # Example 1: Single band structure plot for 3-layer ABC
    print("\n1. Single 3-layer ABC band structure:")
    mlg.set_parameters(N_layers=3, stacking='abc')
    mlg.plot_bands()
    
    # Example 2: Compare ABC vs ABA for same layer number
    print("\n2. Comparing ABC vs ABA for 4 layers:")
    
    # Calculate ABC
    mlg.set_parameters(N_layers=4, stacking='abc')
    E_abc, k_mag = mlg.calculate_bands()
    
    # Calculate ABA
    mlg.set_parameters(stacking='aba')  # Keep N_layers=4
    E_aba, k_mag = mlg.calculate_bands()
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # ABC plot
    ax1.set_title("4-layer ABC")
    nb_abc = E_abc.shape[1]
    mid_abc = [nb_abc//2-1, nb_abc//2]
    for b in range(nb_abc):
        lw = 0.8 if b in mid_abc else 0.5
        ax1.plot(k_mag, E_abc[:,b], linewidth=lw)
    ax1.set_ylabel("Energy (eV)")
    ax1.set_xlim([2.8, 3.1])
    ax1.set_ylim([-0.5, 0.5])
    
    # ABA plot
    ax2.set_title("4-layer ABA")
    nb_aba = E_aba.shape[1]
    mid_aba = [nb_aba//2-1, nb_aba//2]
    for b in range(nb_aba):
        lw = 0.8 if b in mid_aba else 0.5
        ax2.plot(k_mag, E_aba[:,b], linewidth=lw)
    ax2.set_ylabel("Energy (eV)")
    ax2.set_xlim([2.8, 3.1])
    ax2.set_ylim([-0.5, 0.5])
    
    plt.tight_layout()
    plt.show()
    
    # Example 3: Panel comparison for different layer numbers (ABC)
    print("\n3. Panel comparison for ABC stacking (1-8 layers):")
    mlg.set_parameters(stacking='abc')
    mlg.plot_panel_comparison()
    
    # Example 4: Modify parameters and see the effect
    print("\n4. Effect of modified γ1 parameter:")
    
    # Original parameters
    mlg.set_parameters(N_layers=2, stacking='aba', gamma1=0.403)
    E_orig, k_mag = mlg.calculate_bands()
    
    # Modified γ1
    mlg.set_parameters(gamma1=0.6)  # Increase interlayer coupling
    E_mod, k_mag = mlg.calculate_bands()
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original
    ax1.set_title("2-layer ABA (γ₁ = 0.403 eV)")
    for b in range(E_orig.shape[1]):
        ax1.plot(k_mag, E_orig[:,b], linewidth=0.8)
    ax1.set_ylabel("Energy (eV)")
    ax1.set_xlim([2.8, 3.1])
    ax1.set_ylim([-0.5, 0.5])
    
    # Modified
    ax2.set_title("2-layer ABA (γ₁ = 0.600 eV)")
    for b in range(E_mod.shape[1]):
        ax2.plot(k_mag, E_mod[:,b], linewidth=0.8)
    ax2.set_ylabel("Energy (eV)")
    ax2.set_xlim([2.8, 3.1])
    ax2.set_ylim([-0.5, 0.5])
    
    plt.tight_layout()
    plt.show()
    
    # Reset to original value
    mlg.set_parameters(gamma1=0.403)
    
    print("\n5. Direct Hamiltonian access example:")
    # Example of directly using the Hamiltonian
    mlg.set_parameters(N_layers=2, stacking='abc')
    kx, ky = 1/3, 1/3  # K point
    H = mlg.build_hamiltonian(kx, ky)
    eigenvals = sorted(np.linalg.eigvals(H).real)
    print(f"Eigenvalues at K point for 2-layer ABC: {eigenvals}")
    
    print("\nExample completed!")

if __name__ == "__main__":
    import numpy as np
    main()
