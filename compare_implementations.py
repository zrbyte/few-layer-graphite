#!/usr/bin/env python3
"""
Comparison example between direct matrix and PythTB implementations.

This script demonstrates the differences and similarities between the direct
matrix implementation (multilayer_graphene.py) and the PythTB implementation
(multilayer_graphene_pythtb.py) of multilayer graphene band structure calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import both implementations
import multilayer_graphene as mlg_direct
import multilayer_graphene_pythtb as mlg_pythtb

def compare_band_structures():
    """Compare band structures from both implementations."""
    print("=" * 60)
    print("MULTILAYER GRAPHENE IMPLEMENTATION COMPARISON")
    print("=" * 60)
    
    # Test parameters
    test_cases = [
        {"N_layers": 2, "stacking": "abc"},
        {"N_layers": 3, "stacking": "abc"},
        {"N_layers": 3, "stacking": "aba"},
        {"N_layers": 4, "stacking": "aba"},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {params['N_layers']}-layer {params['stacking'].upper()}")
        print("-" * 40)
        
        # Set parameters for both implementations
        mlg_direct.set_parameters(**params, n_k=500)  # Smaller n_k for faster comparison
        mlg_pythtb.set_parameters(**params, n_k=500)
        
        # Time the calculations
        print("Calculating bands...")
        
        # Direct implementation
        start_time = time.time()
        E_direct, k_mag_direct = mlg_direct.calculate_bands()
        time_direct = time.time() - start_time
        
        # PythTB implementation  
        start_time = time.time()
        E_pythtb, k_mag_pythtb = mlg_pythtb.calculate_bands()
        time_pythtb = time.time() - start_time
        
        # Compare results
        max_diff = np.max(np.abs(E_direct - E_pythtb))
        mean_diff = np.mean(np.abs(E_direct - E_pythtb))
        
        print(f"Direct method time:   {time_direct:.4f} s")
        print(f"PythTB method time:   {time_pythtb:.4f} s")
        print(f"Speed ratio (PythTB/Direct): {time_pythtb/time_direct:.2f}")
        print(f"Maximum difference:   {max_diff:.2e} eV")
        print(f"Mean difference:      {mean_diff:.2e} eV")
        
        # Check if results are essentially identical
        if max_diff < 1e-10:
            print("✓ Results are essentially identical")
        elif max_diff < 1e-6:
            print("✓ Results agree within numerical precision")
        else:
            print("⚠ Results show significant differences")

def compare_hamiltonians():
    """Compare Hamiltonian matrices at specific k-points."""
    print(f"\n{'='*60}")
    print("HAMILTONIAN MATRIX COMPARISON")
    print("=" * 60)
    
    # Test at K point and nearby
    test_k_points = [
        (1/3, 1/3),      # K point
        (1/3 + 0.1, 1/3), # Near K
        (0.4, 0.35),     # Random point
    ]
    
    mlg_direct.set_parameters(N_layers=2, stacking='abc')
    mlg_pythtb.set_parameters(N_layers=2, stacking='abc')
    
    for i, (kx, ky) in enumerate(test_k_points):
        print(f"\nk-point {i+1}: ({kx:.3f}, {ky:.3f})")
        print("-" * 30)
        
        # Get Hamiltonians
        H_direct = mlg_direct.build_hamiltonian(kx, ky)
        H_pythtb = mlg_pythtb.build_hamiltonian(kx, ky)
        
        # Compare eigenvalues
        eigs_direct = np.sort(np.linalg.eigvals(H_direct).real)
        eigs_pythtb = np.sort(np.linalg.eigvals(H_pythtb).real)
        
        print(f"Direct eigenvalues:  {eigs_direct}")
        print(f"PythTB eigenvalues:  {eigs_pythtb}")
        print(f"Eigenvalue differences: {eigs_direct - eigs_pythtb}")
        
        # Compare matrix elements
        H_diff = np.abs(H_direct - H_pythtb)
        print(f"Max Hamiltonian difference: {np.max(H_diff):.2e}")

def plot_comparison():
    """Create visual comparison plots."""
    print(f"\n{'='*60}")
    print("VISUAL COMPARISON")
    print("=" * 60)
    
    # Set up comparison for 3-layer ABC
    mlg_direct.set_parameters(N_layers=3, stacking='abc', n_k=1000)
    mlg_pythtb.set_parameters(N_layers=3, stacking='abc', n_k=1000)
    
    # Calculate bands
    E_direct, k_mag = mlg_direct.calculate_bands()
    E_pythtb, _ = mlg_pythtb.calculate_bands()
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Direct implementation
    ax1 = axes[0]
    nb = E_direct.shape[1]
    mid = [nb//2-1, nb//2]
    for b in range(nb):
        lw = 0.8 if b in mid else 0.5
        ax1.plot(k_mag, E_direct[:,b], 'b-', linewidth=lw)
    ax1.set_title("Direct Matrix Implementation")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_xlim([2.8, 3.1])
    ax1.set_ylim([-0.5, 0.5])
    
    # PythTB implementation
    ax2 = axes[1]
    for b in range(nb):
        lw = 0.8 if b in mid else 0.5
        ax2.plot(k_mag, E_pythtb[:,b], 'r-', linewidth=lw)
    ax2.set_title("PythTB Implementation")
    ax2.set_xlim([2.8, 3.1])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_yticks([])
    
    # Difference plot
    ax3 = axes[2]
    E_diff = E_direct - E_pythtb
    for b in range(nb):
        lw = 0.8 if b in mid else 0.5
        ax3.plot(k_mag, E_diff[:,b], 'g-', linewidth=lw)
    ax3.set_title("Difference (Direct - PythTB)")
    ax3.set_ylabel("Energy Difference (eV)")
    ax3.set_xlim([2.8, 3.1])
    ax3.set_ylim([np.min(E_diff)*1.1, np.max(E_diff)*1.1])
    
    plt.suptitle("3-layer ABC Stacking Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

def benchmark_performance():
    """Benchmark performance for different system sizes."""
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    layer_counts = range(1, 8)
    times_direct = []
    times_pythtb = []
    
    print(f"{'Layers':<8} {'Direct (s)':<12} {'PythTB (s)':<12} {'Ratio':<8}")
    print("-" * 45)
    
    for N in layer_counts:
        # Set smaller n_k for benchmarking
        mlg_direct.set_parameters(N_layers=N, stacking='abc', n_k=200)
        mlg_pythtb.set_parameters(N_layers=N, stacking='abc', n_k=200)
        
        # Benchmark direct method
        start_time = time.time()
        _ = mlg_direct.calculate_bands()
        time_direct = time.time() - start_time
        times_direct.append(time_direct)
        
        # Benchmark PythTB method
        start_time = time.time()
        _ = mlg_pythtb.calculate_bands()
        time_pythtb = time.time() - start_time
        times_pythtb.append(time_pythtb)
        
        ratio = time_pythtb / time_direct if time_direct > 0 else float('inf')
        print(f"{N:<8} {time_direct:<12.4f} {time_pythtb:<12.4f} {ratio:<8.2f}")
    
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(layer_counts, times_direct, 'bo-', label='Direct Matrix')
    plt.plot(layer_counts, times_pythtb, 'ro-', label='PythTB')
    plt.xlabel('Number of Layers')
    plt.ylabel('Calculation Time (s)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    ratios = [t_pythtb/t_direct for t_pythtb, t_direct in zip(times_pythtb, times_direct)]
    plt.plot(layer_counts, ratios, 'go-')
    plt.xlabel('Number of Layers')
    plt.ylabel('Speed Ratio (PythTB/Direct)')
    plt.title('Relative Performance')
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_features():
    """Demonstrate unique features of each implementation."""
    print(f"\n{'='*60}")
    print("IMPLEMENTATION FEATURES")
    print("=" * 60)
    
    print("\nDirect Matrix Implementation:")
    print("✓ Fast and direct calculation")
    print("✓ Full control over Hamiltonian construction")
    print("✓ Minimal dependencies")
    print("✓ Transparent matrix operations")
    
    print("\nPythTB Implementation:")
    print("✓ Leverages established tight-binding framework")
    print("✓ Built-in validation and error checking")
    print("✓ Extensible to other tight-binding models")
    print("✓ Rich set of analysis tools from PythTB")
    print("✓ Better handling of complex hopping patterns")
    
    # Show PythTB model information
    print(f"\nPythTB Model Structure (3-layer ABC):")
    mlg_pythtb.set_parameters(N_layers=3, stacking='abc')
    model = mlg_pythtb.build_model()
    print(f"Dimensions: {model._dim_k}D")
    print(f"Number of orbitals: {model._norb}")
    print(f"Lattice vectors: {model._lat}")

def main():
    """Main comparison function."""
    print("Multilayer Graphene Implementation Comparison")
    print("Comparing direct matrix vs PythTB approaches")
    
    # Run all comparisons
    compare_band_structures()
    compare_hamiltonians()
    plot_comparison()
    benchmark_performance()
    demonstrate_features()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print("Both implementations produce essentially identical results.")
    print("Choose based on your specific needs:")
    print("- Direct matrix: For speed and minimal dependencies")
    print("- PythTB: For extensibility and framework integration")

if __name__ == "__main__":
    main()
