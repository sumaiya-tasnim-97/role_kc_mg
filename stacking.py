"""
Base Stacking Comparison in RNA Loops Using MDTraj

This script analyzes base stacking interactions within and between two RNA loop regions across
two molecular dynamics (MD) trajectories using MDTraj. It computes stacking occurrences per
residue pair, calculates stacking frequencies, and generates a comparative bar plot.

Requirements:
-------------
- Python packages: mdtraj, numpy, matplotlib
- Input files:
    • Two MD trajectory files (.dcd)
    • Corresponding topology files (.prmtop)
    • Each must represent the same system but under different conditions (e.g., with/without Mg²⁺)
- Configured loop residue ranges (loop1 and loop2)
- Output: A bar plot image comparing base stacking frequencies between the two trajectories

Stacking is identified based on inter-base distance (< 4.5 Å) and angle between base-plane
normals (< 30°). The results are averaged over trajectory frames and plotted side by side.
"""


import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

# === USER CONFIG ===
traj_info = [
    ('kc_mg_0_uw.dcd', 'kc_mg_0.prmtop', 'JM_output_kc_mg_0_310-kmeans-CS1.pdb.pdb'),
    ('kc_mg_8_uw.dcd', 'kc_mg_8.prmtop', 'JM_output_kc_mg_8-kmeans-CS1.pdb.pdb')
]
loop1 = list(range(17, 28))  # Residues 17–27
loop2 = list(range(58, 69))  # Residues 58–68
loop_residues = loop1 + loop2

# === FUNCTIONS ===

def get_resseq_map(traj, residues):
    resseq_map = {}
    for res_id in residues:
        resseq_map[res_id] = traj.topology.residue(res_id).resSeq
    return resseq_map

def compute_base_stacking(traj, residue_pairs):
    stacking_results = {f"{r1}-{r2}": [] for r1, r2 in residue_pairs}

    for frame in traj:
        for r1, r2 in residue_pairs:
            try:
                res1_atoms = [a.index for a in frame.topology.residue(r1).atoms
                              if a.name in ["C2", "C4", "C5", "C6", "C8", "N1", "N3", "N7", "N9"]]
                res2_atoms = [a.index for a in frame.topology.residue(r2).atoms
                              if a.name in ["C2", "C4", "C5", "C6", "C8", "N1", "N3", "N7", "N9"]]

                if len(res1_atoms) < 3 or len(res2_atoms) < 3:
                    stacking_results[f"{r1}-{r2}"].append(False)
                    continue

                coords1 = frame.xyz[0][res1_atoms]
                coords2 = frame.xyz[0][res2_atoms]

                normal1 = np.cross(coords1[1] - coords1[0], coords1[2] - coords1[0])
                normal2 = np.cross(coords2[1] - coords2[0], coords2[2] - coords2[0])

                normal1 /= np.linalg.norm(normal1)
                normal2 /= np.linalg.norm(normal2)

                angle = np.degrees(np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0)))
                centroid1 = np.mean(coords1, axis=0)
                centroid2 = np.mean(coords2, axis=0)
                distance = np.linalg.norm(centroid1 - centroid2)

                stacked = (distance < 4.5) and (angle < 30)
                stacking_results[f"{r1}-{r2}"].append(stacked)
            except:
                stacking_results[f"{r1}-{r2}"].append(False)

    return stacking_results

def compute_stacking_frequencies(stacking_dict):
    freq_dict = {}
    for pair, stack_list in stacking_dict.items():
        freq = np.sum(stack_list) / len(stack_list)
        freq_dict[pair] = freq
    return freq_dict

def plot_stacking_comparison(freq_dict1, freq_dict2, label1, label2, save_path):
    all_pairs = sorted(set(freq_dict1.keys()) | set(freq_dict2.keys()))
    freqs1 = [freq_dict1.get(p, 0.0) for p in all_pairs]
    freqs2 = [freq_dict2.get(p, 0.0) for p in all_pairs]

    x = np.arange(len(all_pairs))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, freqs1, width, label=label1)
    plt.bar(x + width/2, freqs2, width, label=label2)

    plt.xticks(x, all_pairs, rotation=45, ha='right')
    plt.ylabel('Stacking Frequency')
    plt.title('Base Stacking Comparison Between Trajectories')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def get_residue_pairs_for_stacking(loop1, loop2):
    intra_loop1 = [(loop1[i], loop1[i + 1]) for i in range(len(loop1) - 1)]
    intra_loop2 = [(loop2[i], loop2[i + 1]) for i in range(len(loop2) - 1)]
    inter_loop = list(zip(loop1, loop2))
    return intra_loop1 + intra_loop2 + inter_loop

# === MAIN EXECUTION ===

residue_pairs = get_residue_pairs_for_stacking(loop1, loop2)

traj1 = md.load(traj_info[0][0], top=traj_info[0][1], stride=100)
traj2 = md.load(traj_info[1][0], top=traj_info[1][1], stride=100)

stacking_dict_traj1 = compute_base_stacking(traj1, residue_pairs)
stacking_dict_traj2 = compute_base_stacking(traj2, residue_pairs)

freqs1 = compute_stacking_frequencies(stacking_dict_traj1)
freqs2 = compute_stacking_frequencies(stacking_dict_traj2)

plot_stacking_comparison(freqs1, freqs2, label1='traj1', label2='traj2', save_path='stacking_comparison.png')

print("\nAll done!")

