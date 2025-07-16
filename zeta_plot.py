"""
Zeta (ζ) Torsion Analysis for RNA Loop Residues using MDTraj

Description:
    This script computes and visualizes the ζ (zeta) backbone torsion angle time series 
    and distributions for selected loop residues in RNA, across multiple MD trajectories.
    It compares dynamic ζ angle behavior with reference values from centroid structures,
    providing insight into structural rigidity or flexibility at the residue level.

Torsion Definition:
    The ζ angle is defined over four atoms: C3' (i), O3' (i), P (i+1), O5' (i+1)
    This script correctly accounts for atom offsets across residues.

Key Features:
    - Computes per-residue ζ angle time series (in degrees) from MD trajectories.
    - Strides trajectory frames for performance (adjustable via `frame_stride`).
    - Estimates circular mean (μ) and circular standard deviation (σ) for each distribution.
    - Overlays reference ζ values from centroid PDB structures.
    - Outputs both time series and distribution plots per residue.

Required Inputs:
    - `traj_info`: A list of tuples with trajectory (.dcd), topology (.prmtop), 
      and reference centroid structure (.pdb) for each simulation.
    - `traj_labels`: Labels for each trajectory to use in legends.
    - Loop residues: Lists of integer residue indices defining loop regions (adjustable).

Dependencies:
    - MDTraj
    - NumPy
    - SciPy (for circular statistics)
    - Matplotlib
    - Seaborn

Outputs:
    - Directory: `zeta_mdtraj_plots/`
        • `zeta_timeseries_resXX.png`: ζ angle time series for each loop residue.
        • `zeta_distribution_resXX.png`: ζ angle distributions (KDE) for each residue.
    - Plots include reference ζ values as dashed lines and μ/σ in labels.

Configuration Parameters:
    - `frame_stride`: Controls frame sampling frequency (default: 10).
    - `loop1_residues`, `loop2_residues`: Specify which residues to analyze.
    - `output_dir`: Target directory for all saved plots.

Usage:
    - Run the script directly in a Python environment.
    - Ensure topology and trajectories are properly aligned and formatted.
    - Modify the residue ranges or ζ atom definitions as needed for non-standard systems.

Author Notes:
    - The ζ angle is a sensitive metric for capturing local backbone fluctuations.
    - Per-residue plots help differentiate between rigid and flexible sites in structured loops.
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

output_dir = "zeta_mdtraj_plots"
os.makedirs(output_dir, exist_ok=True)
frame_stride = 10

# === Zeta torsion atom definition ===
zeta_atoms = ("C3'", "O3'", "P", "O5'")

def get_zeta_indices(topology, resid):
    indices = []
    for j, name in enumerate(zeta_atoms):
        offset = 1 if j >= 2 else 0  # P and O5' are from next residue
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

# === Load reference zeta values ===
ref_zeta_values = {}
for (_, top_file, pdb_file), label in zip(traj_info, traj_labels):
    ref = md.load(pdb_file, top=top_file)
    angles = {}
    for resid in loop_residues:
        inds = get_zeta_indices(ref.topology, resid)
        if inds:
            try:
                angle = md.compute_dihedrals(ref, [inds])[0, 0]
                angles[resid] = np.rad2deg(angle)
            except:
                continue
    ref_zeta_values[label] = angles

# === Main plot loop ===
ref_colors = {'kc_mg_0': 'red', 'kc_mg_8': 'green'}
for resid in loop_residues:
    plt.figure(figsize=(10, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        inds = get_zeta_indices(traj.topology, resid)
        if not inds:
            continue
        zeta_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        zeta_deg = np.rad2deg(zeta_rad)

        zeta_deg_strided = zeta_deg[::frame_stride]
        frames = np.arange(len(zeta_deg))[::frame_stride]

        mu = np.rad2deg(circmean(zeta_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(zeta_rad, high=np.pi, low=-np.pi))

        plt.plot(frames, zeta_deg_strided, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", lw=1.2)

    # Reference lines
    for label in traj_labels:
        if resid in ref_zeta_values[label]:
            val = ref_zeta_values[label][resid]
            plt.axhline(val, linestyle='--', color=ref_colors[label], alpha=0.6)
            plt.text(frames[-1] + 5, val, f'ref: {label}', fontsize=8, color=ref_colors[label], va='center')

    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Zeta (degrees)", fontsize=12)
    plt.title(f"Zeta Torsion Time Series — Residue {resid}", fontsize=14)
    plt.legend()
    plt.ylim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zeta_timeseries_res{resid}.png")
    plt.close()

    # === Distribution plot ===
    plt.figure(figsize=(6, 4), dpi=300)
    for (traj_file, top_file, _), label in zip(traj_info, traj_labels):
        traj = md.load(traj_file, top=top_file)
        inds = get_zeta_indices(traj.topology, resid)
        if not inds:
            continue
        zeta_rad = md.compute_dihedrals(traj, [inds])[:, 0]
        zeta_deg = np.rad2deg(zeta_rad)

        mu = np.rad2deg(circmean(zeta_rad, high=np.pi, low=-np.pi))
        sigma = np.rad2deg(circstd(zeta_rad, high=np.pi, low=-np.pi))

        sb.kdeplot(zeta_deg, label=f"{label} (\u03bc={mu:.1f}°, σ={sigma:.1f}°)", fill=True, alpha=0.3)

    for label in traj_labels:
        if resid in ref_zeta_values[label]:
            val = ref_zeta_values[label][resid]
            plt.axvline(val, linestyle='--', color=ref_colors[label], alpha=0.6, label=f'ref: {label}')

    plt.xlabel("Zeta (degrees)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Zeta Torsion Distribution — Residue {resid}", fontsize=14)
    plt.legend()
    plt.xlim(-180, 180)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zeta_distribution_res{resid}.png")
    plt.close()

