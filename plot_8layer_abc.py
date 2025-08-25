#!/usr/bin/env python3
"""Plot 8-layer ABC graphite band structure including next-nearest interlayer hoppings.

Generates band structure along the Γ → K → Γ path. Two sets of
hoppings are compared: one with only γ₀ and γ₁ (red dashed, perfectly flat
band) and one with the full Grüneis parameters including γ₂–γ₅ (black
solid) which induces a slight downward dispersion of the nominally flat band.
"""
import matplotlib.pyplot as plt
import multilayer_graphene as mlg


def main() -> None:
    """Compute and plot the band structure for eight-layer ABC graphite."""
    # Calculate with only nearest neighbour hoppings (γ₀ and γ₁)
    mlg.set_parameters(N_layers=8, stacking='abc',
                       gamma2=0.0, gamma3=0.0, gamma4=0.0, gamma5=0.0)
    E_nn, k_mag = mlg.calculate_bands(path_type='gkg')

    # Calculate with full Grüneis parameters including next-nearest hoppings
    mlg.set_parameters(gamma2=-0.025, gamma3=0.274, gamma4=0.143, gamma5=0.030)
    E_full, _ = mlg.calculate_bands(path_type='gkg')

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot full-hopping bands in black
    for b in range(E_full.shape[1]):
        ax.plot(k_mag, E_full[:, b], color='black', linewidth=0.8)

    # Overlay nearest-neighbour-only bands (show flatness) in dashed red
    for b in range(E_nn.shape[1]):
        ax.plot(k_mag, E_nn[:, b], color='red', linestyle='--', linewidth=0.6)

    ax.set_xlabel('k along Γ → K → Γ')
    ax.set_ylabel('Energy (eV)')
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlim(k_mag.min(), k_mag.max())
    ax.set_title('8-layer ABC graphite around K')
    plt.tight_layout()

    # Save figure so it can be viewed in headless environments
    fig.savefig('abc8_bands.png', dpi=150)
    print('Saved band structure figure to abc8_bands.png')

    plt.show()


if __name__ == '__main__':
    main()
