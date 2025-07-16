"""
Chi1 (χ₁) Torsion Analysis for RNA Loop Residues using MDTraj

Description:
    This script calculates and visualizes the χ₁ torsion angle (syn/anti glycosidic bond orientation)
    for specified RNA loop residues across multiple MD trajectories.
    It compares dynamic χ₁ behavior to reference values from centroid (e.g., cluster center) structures,
    enabling insight into base orientation stability and flexibility.

Torsion Definition:
    χ₁ torsion depends on the base type:
        • Purines (A, G): χ₁ = O4'–C1'–N9–C4
        • Pyrimidines (C, U): χ₁ = O4'–C1'–N1–C2
    The script automatically assigns the correct atom indices per residue.

Key Features:
    - Computes χ₁ angle time series (in degrees) over MD trajectories.
    - Applies optional frame striding for efficient analysis.
    - Estimates per-residue circular mean (μ) and circular standard deviation (σ).
    - Overlays χ₁ reference values from PDB structures.
    - Produces time series and density plots for each residue analyzed.

Inputs:
    - `traj_info`: List of tuples (trajectory.dcd, topology.prmtop, reference.pdb).
    - `traj_labels`: Short names for each simulation used in plot legends.
    - `loop_residues`: Residue indices for RNA loops to be analyzed.

Dependencies:
    - MDTraj
    - NumPy
    - SciPy (for circular statistics)
    - Seaborn
    - Matplotlib

Outputs:
    - Directory: `chi1_mdtraj_plots/`
        • `chi1_timeseries_resXX.png`: χ₁ angle time series per residue.
        • `chi1_distribution_resXX.png`: χ₁ KDE plot showing angle distributions.
    - Reference χ₁ values are shown as dashed lines per simulation.

Configuration:
    - `frame_stride`: Frame interval for downsampling (default: 10).
    - `loop1_residues`, `loop2_residues`: Index ranges for each RNA loop.
    - `output_dir`: Path for saving plots.

Usage:
    - Make sure `.dcd`, `.prmtop`, and reference `.pdb` files are aligned and compatible.
    - Adjust residue indices and striding as appropriate for your system.
    - Run the script in a Python environment to generate per-residue χ₁ plots.

Scientific Context:
    The χ₁ angle reflects base orientation (syn vs anti) and is a critical torsion
    in describing RNA base dynamics. Comparing χ₁ distributions between conditions
    can inform on loop flexibility, conformational preferences, and the influence
    of factors like Mg²⁺ or mutations.

Author Notes:
    The automated handling of purine vs pyrimidine atom definitions makes the script
    broadly applicable across RNA systems. Output plots are suitable for figure inclusion
    in manuscripts or supplementary material.
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

output_dir = "chi1_mdtraj_plots"
os.makedirs(output_dir, exist_ok=True)
frame_stride = 10

# === Function to find chi1 torsion atom indices ===
def find_chi1_torsions(topology):
    chi1_torsions = {}
    for res in topology.residues:
        atoms = {atom.name: atom.index for atom in res.atoms}
        if all(name in atoms for name in ["O4'", "C1'", "N9", "C4"]):
            chi1_torsions[res.resSeq] = (atoms["O4'"], atoms["C1'"], atoms["N9"], atoms["C4"])
        elif all(name in atoms for name in ["O4'", "C1'", "N1", "C2"]):
            chi1_torsions[res.resSeq] = (atoms["O4'"], atoms["C1'"], atoms["N1"], atoms["C2"])
    return chi1_torsions

# === Load reference chi1 values ===
ref_chi_values = {}
for (_, top_file, pdb_file), label in zip(traj_info, traj_labels):
    ref = md.load(pdb_file, top=top_file)
    chi1_indices = find_chi1_torsions(ref.topology)
    angles = {}
    for resid, inds in chi1_indices.items():
        angle = md.compute_dihedrals(ref, [inds])[0, 0]
        angles[resid] = np.rad2deg(angle)
    ref_chi_values[label] = angles

# === Main plot loop ===
for resid in loop_residues:
    plt.figure(figsize=(10, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        chi1_indices = find_chi1_torsions(traj.topology)
        if resid not in chi1_indices:
            continue
        inds = chi1_indices[resid]
        chi1_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        chi1_deg = np.rad2deg(chi1_rad)

        chi1_deg_strided = chi1_deg[::frame_stride]
        frames = np.arange(len(chi1_deg))[::frame_stride]

        mu = np.rad2deg(circmean(chi1_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(chi1_rad, high=np.pi, low=-np.pi))

        plt.plot(frames, chi1_deg_strided, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", lw=1.2)

    # Reference lines
    ref_colors = {'kc_mg_0': 'red', 'kc_mg_8': 'green'}
    for label in traj_labels:
     if resid in ref_chi_values[label]:
        val = ref_chi_values[label][resid]
        plt.axhline(val, linestyle='--', color=ref_colors[label], alpha=0.6)
        plt.text(frames[-1] + 5, val, f'ref: {label}', fontsize=8, color=ref_colors[label], va='center')

    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("\u03c71 (degrees)", fontsize=12)
    plt.title(f"\u03c71 Time Series — Residue {resid}", fontsize=14)
    plt.legend()
    plt.ylim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chi1_timeseries_res{resid}.png")
    plt.close()

    # === Distribution plot ===
    plt.figure(figsize=(6, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        chi1_indices = find_chi1_torsions(traj.topology)
        if resid not in chi1_indices:
            continue
        inds = chi1_indices[resid]
        chi1_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        chi1_deg = np.rad2deg(chi1_rad)

        mu = np.rad2deg(circmean(chi1_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(chi1_rad, high=np.pi, low=-np.pi))

        sb.kdeplot(chi1_deg, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", fill=True, alpha=0.3)

    for label in traj_labels:
     if resid in ref_chi_values[label]:
        val = ref_chi_values[label][resid]
        plt.axvline(val, linestyle='--', color=ref_colors[label], alpha=0.6, label=f'ref: {label}')

    plt.xlabel("\u03c71 (degrees)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"\u03c71 Distribution — Residue {resid}", fontsize=14)
    plt.legend()
    plt.xlim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chi1_distribution_res{resid}.png")
    plt.close()
