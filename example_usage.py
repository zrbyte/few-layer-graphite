#!/usr/bin/env python3
"""
Short tour of multilayer_graphene: bands, Hamiltonians and density of states
for the stacking polytypes of few-layer graphene.
"""
import numpy as np
import matplotlib.pyplot as plt

import multilayer_graphene as mlg


def main():
    mlg.get_info()

    # 1. Band structures of the three tetralayer polytypes along p_x through K
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, seq in zip(axes, ["ABAB", "ABCA", "ABCB"]):
        E, k = mlg.calculate_bands(stacking_type=seq, path_type="px")
        mlg.plot_bands(E, k, stacking_type=seq, path_type="px", xlim=(-0.2, 0.2),
                       ylim=(-0.8, 0.8), ax=ax, show=False)
    fig.tight_layout()
    plt.show()

    # 2. Layer-number dependence of rhombohedral stacking
    mlg.set_parameters(stacking="abc")
    mlg.plot_panel_comparison(range(1, 7), xlim=(-0.2, 0.2))

    # 3. Band energies at the valley centre: dimer edges at Delta' +- gamma1
    K = mlg.K_point()
    H = mlg.hamiltonian(K[0], K[1], stacking_type="ABCA")
    print("ABCA band energies at K (eV):", np.round(np.linalg.eigvalsh(H), 4))

    # 4. Density of states (states per eV per cm^2, spin included) like Fig. 1 of the paper
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, seq in zip(axes, ["ABAB", "ABCA", "ABCB"]):
        mlg.plot_dos((-0.8, 0.8), stacking_type=seq, ax=ax, show=False)
    fig.tight_layout()
    plt.show()

    # 5. Lattice model versus the paper's k.p model close to K
    p = np.array([0.02, 0.01])
    E_tb = np.linalg.eigvalsh(mlg.hamiltonian(K[0] + p[0], K[1] + p[1], stacking_type="ABCB"))
    E_kp = np.linalg.eigvalsh(mlg.kp_hamiltonian(p[0], p[1], stacking_type="ABCB"))
    print("ABCB at |p| = 0.022 1/A: max |E_tb - E_kp| = %.2e eV" % np.abs(E_tb - E_kp).max())


if __name__ == "__main__":
    main()
