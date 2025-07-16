
"""
Alpha (α) Backbone Torsion Analysis for RNA Loop Residues using MDTraj

Description:
    This script analyzes the α torsion angle (O3'–P–O5'–C5') across defined RNA loop residues 
    in multiple MD trajectories. It compares the dynamic α behavior to reference values
    extracted from static reference structures (e.g., cluster centroids or minimized conformations).

Torsion Definition:
    • α = O3'(i−1) – P(i) – O5'(i) – C5'(i)
    The O3' atom belongs to the previous residue (i−1), while the remaining atoms are from residue i.
    The script handles this offset automatically.

Key Features:
    - Identifies α dihedral indices from topology using atom name mapping.
    - Computes α angle time series and corresponding strided data for visualization.
    - Calculates circular mean (μ) and standard deviation (σ) per trajectory and residue.
    - Plots both time evolution and angle distributions (with KDE) per residue.
    - Annotates plots with dashed reference lines from static structures.

Inputs:
    - `traj_info`: List of tuples in the form (trajectory.dcd, topology.prmtop, reference.pdb).
    - `traj_labels`: Human-readable labels for simulations, used in plot legends.
    - `loop_residues`: List of residue indices (e.g., hairpin/kissing loop residues).

Dependencies:
    - MDTraj
    - NumPy
    - SciPy (for circular statistics)
    - Seaborn
    - Matplotlib

Outputs:
    - Directory: `alpha_mdtraj_plots/`
        • `alpha_timeseries_resXX.png`: Time series plot of α angle per residue.
        • `alpha_distribution_resXX.png`: KDE plot of α angle distribution per residue.

Configuration:
    - `frame_stride`: Frequency at which frames are sampled (to reduce noise or speed up analysis).
    - `loop1_residues`, `loop2_residues`: Specify target RNA loop regions.

Usage:
    - Ensure all `.dcd`, `.prmtop`, and `.pdb` files are aligned and structurally compatible.
    - Adjust `loop_residues` and `frame_stride` as needed.
    - Execute the script in a Python environment to generate all plots automatically.

Scientific Context:
    The α torsion is part of the RNA backbone and contributes to overall helical geometry.
    Analyzing fluctuations in α can provide insight into local flexibility, backbone constraints,
    or the effects of ions like Mg²⁺ on the loop conformation.

Author Notes:
    This script is modular, extendable to other backbone torsions (β, γ, etc.), and outputs
    figures suitable for publication or further statistical comparison between simulation conditions.
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

output_dir = "alpha_mdtraj_plots"
os.makedirs(output_dir, exist_ok=True)
frame_stride = 10

# === Alpha torsion atom definition ===
alpha_atoms = ("O3'", 'P', "O5'", "C5'")

def get_alpha_indices(topology, resid):
    indices = []
    for j, name in enumerate(alpha_atoms):
        offset = -1 if j == 0 else 0  # O3' comes from previous residue
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

# === Load reference alpha values ===
ref_alpha_values = {}
for (_, top_file, pdb_file), label in zip(traj_info, traj_labels):
    ref = md.load(pdb_file, top=top_file)
    angles = {}
    for resid in loop_residues:
        inds = get_alpha_indices(ref.topology, resid)
        if inds:
            try:
                angle = md.compute_dihedrals(ref, [inds])[0, 0]
                angles[resid] = np.rad2deg(angle)
            except:
                continue
    ref_alpha_values[label] = angles

# === Main plot loop ===
for resid in loop_residues:
    plt.figure(figsize=(10, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        inds = get_alpha_indices(traj.topology, resid)
        if not inds:
            continue
        alpha_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        alpha_deg = np.rad2deg(alpha_rad)

        alpha_deg_strided = alpha_deg[::frame_stride]
        frames = np.arange(len(alpha_deg))[::frame_stride]

        mu = np.rad2deg(circmean(alpha_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(alpha_rad, high=np.pi, low=-np.pi))

        plt.plot(frames, alpha_deg_strided, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", lw=1.2)

    # Reference lines
    ref_colors = {'kc_mg_0': 'red', 'kc_mg_8': 'green'}
    for label in traj_labels:
     if resid in ref_alpha_values[label]:
        val = ref_alpha_values[label][resid]
        plt.axhline(val, linestyle='--', color=ref_colors[label], alpha=0.6)
        plt.text(frames[-1] + 5, val, f'ref: {label}', fontsize=8, color=ref_colors[label], va='center')

    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Alpha (degrees)", fontsize=12)
    plt.title(f"Alpha Torsion Time Series — Residue {resid}", fontsize=14)
    plt.legend()
    plt.ylim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/alpha_timeseries_res{resid}.png")
    plt.close()

    # === Distribution plot ===
    plt.figure(figsize=(6, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        inds = get_alpha_indices(traj.topology, resid)
        if not inds:
            continue
        alpha_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        alpha_deg = np.rad2deg(alpha_rad)

        mu = np.rad2deg(circmean(alpha_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(alpha_rad, high=np.pi, low=-np.pi))

        sb.kdeplot(alpha_deg, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", fill=True, alpha=0.3)

    for label in traj_labels:
     if resid in ref_alpha_values[label]:
        val = ref_alpha_values[label][resid]
        plt.axvline(val, linestyle='--', color=ref_colors[label], alpha=0.6, label=f'ref: {label}')

    plt.xlabel("Alpha (degrees)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Alpha Torsion Distribution — Residue {resid}", fontsize=14)
    plt.legend()
    plt.xlim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/alpha_distribution_res{resid}.png")
    plt.close()

