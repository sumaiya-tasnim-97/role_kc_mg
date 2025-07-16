"""
Epsilon (ε) Backbone Torsion Analysis for RNA Loop Residues using MDTraj

Description:
    This script computes and visualizes the ε torsion angle (C4′–C3′–O3′–P) for selected RNA residues 
    across molecular dynamics (MD) trajectories. It provides per-residue time series plots and angle 
    distribution plots for different trajectory conditions (e.g., ionic concentrations).

Torsion Definition:
    • ε = C4′(i) – C3′(i) – O3′(i) – P(i+1)
    This torsion spans residue i and its following residue (i+1). The script automatically handles the 
    offset for accessing atoms from adjacent residues.

Key Features:
    - Automatically identifies torsion atom indices for each residue.
    - Computes ε dihedral angles from trajectories and reference structures.
    - Produces two types of plots per residue:
        1. Time series of ε angles across frames (with circular mean and standard deviation).
        2. KDE-based angular distributions with overlaid reference values.
    - Reference values are extracted from PDBs (e.g., representative cluster structures).

Inputs:
    - `traj_info`: List of tuples (trajectory file, topology file, reference pdb file).
    - `traj_labels`: Human-readable names for each trajectory condition (used in plots).
    - `loop_residues`: List of target residues (e.g., residues in RNA kissing loops).

Dependencies:
    - MDTraj
    - NumPy
    - SciPy (circular statistics)
    - Matplotlib
    - Seaborn

Outputs:
    - Directory: `epsilon_mdtraj_plots/`
        • `epsilon_timeseries_resXX.png`: Time evolution of ε angle for residue XX.
        • `epsilon_distribution_resXX.png`: Angular distribution of ε angle for residue XX.

Configurable Parameters:
    - `frame_stride`: Plotting interval used to reduce overplotting and smooth visuals.
    - `loop1_residues`, `loop2_residues`: Define custom loop or region of interest.

Scientific Context:
    The ε torsion influences RNA backbone flexibility, particularly in regions undergoing conformational 
    transitions like loop-loop kissing interactions. Quantifying ε fluctuations enables characterization 
    of local rigidity, backbone disruptions, and the influence of ionic conditions on RNA architecture.

Use Case:
    • Comparing conformational dynamics between apo and Mg²⁺-bound systems.
    • Identifying residues with altered ε distributions under different structural states.

Author Notes:
    This script is part of a modular torsion analysis suite and follows the same structure as alpha, beta, 
    and zeta analysis scripts. Output figures are designed for direct use in publication or presentation.
"""



import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import circmean, circstd
import os
import warnings

warnings.filterwarnings('ignore')

# === USER CONFIG ===
traj_info = [
    ('kc_mg_0_uw.dcd', 'kc_mg_0.prmtop', 'JM_output_kc_mg_0_310-kmeans-CS1.pdb'),
    ('kc_mg_8_uw.dcd', 'kc_mg_8.prmtop', 'JM_output_kc_mg_8-kmeans-CS1.pdb')
]
traj_labels = ['kc_mg_0', 'kc_mg_8']
loop1_residues = list(range(17, 28))
loop2_residues = list(range(58, 69))
loop_residues = loop1_residues + loop2_residues

output_dir = "epsilon_mdtraj_plots"
os.makedirs(output_dir, exist_ok=True)
frame_stride = 10

# === Epsilon torsion atom definition ===
epsilon_atoms = ("C4'", "C3'", "O3'", "P")

def get_epsilon_indices(topology, resid):
    indices = []
    for j, name in enumerate(epsilon_atoms):
        offset = 1 if j == 3 else 0  # P comes from the next residue
        res_target = resid + offset
        if 0 <= res_target < topology.n_residues:
            try:
                atom = [a.index for a in topology.residue(res_target).atoms if a.name == name][0]
                indices.append(atom)
            except:
                return None
        else:
            return None
    return tuple(indices)

# === Load reference epsilon values ===
ref_epsilon_values = {}
for (_, top_file, pdb_file), label in zip(traj_info, traj_labels):
    ref = md.load(pdb_file, top=top_file)
    angles = {}
    for resid in loop_residues:
        inds = get_epsilon_indices(ref.topology, resid)
        if inds:
            try:
                angle = md.compute_dihedrals(ref, [inds])[0, 0]
                angles[resid] = np.rad2deg(angle)
            except:
                continue
    ref_epsilon_values[label] = angles

# === Main plot loop ===
ref_colors = {'kc_mg_0': 'red', 'kc_mg_8': 'green'}
for resid in loop_residues:
    plt.figure(figsize=(10, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        inds = get_epsilon_indices(traj.topology, resid)
        if not inds:
            continue
        epsilon_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        epsilon_deg = np.rad2deg(epsilon_rad)

        epsilon_deg_strided = epsilon_deg[::frame_stride]
        frames = np.arange(len(epsilon_deg))[::frame_stride]

        mu = np.rad2deg(circmean(epsilon_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(epsilon_rad, high=np.pi, low=-np.pi))

        plt.plot(frames, epsilon_deg_strided, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", lw=1.2)

    # Reference lines
    for label in traj_labels:
        if resid in ref_epsilon_values[label]:
            val = ref_epsilon_values[label][resid]
            plt.axhline(val, linestyle='--', color=ref_colors[label], alpha=0.6)
            plt.text(frames[-1] + 5, val, f'ref: {label}', fontsize=8, color=ref_colors[label], va='center')

    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Epsilon (degrees)", fontsize=12)
    plt.title(f"Epsilon Torsion Time Series — Residue {resid}", fontsize=14)
    plt.legend()
    plt.ylim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epsilon_timeseries_res{resid}.png")
    plt.close()

    # === Distribution plot ===
    plt.figure(figsize=(6, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        inds = get_epsilon_indices(traj.topology, resid)
        if not inds:
            continue
        epsilon_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        epsilon_deg = np.rad2deg(epsilon_rad)

        mu = np.rad2deg(circmean(epsilon_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(epsilon_rad, high=np.pi, low=-np.pi))

        sb.kdeplot(epsilon_deg, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", fill=True, alpha=0.3)

    for label in traj_labels:
        if resid in ref_epsilon_values[label]:
            val = ref_epsilon_values[label][resid]
            plt.axvline(val, linestyle='--', color=ref_colors[label], alpha=0.6, label=f'ref: {label}')

    plt.xlabel("Epsilon (degrees)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Epsilon Torsion Distribution — Residue {resid}", fontsize=14)
    plt.legend()
    plt.xlim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epsilon_distribution_res{resid}.png")
    plt.close()

