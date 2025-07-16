"""
This script computes and compares the root-mean-square fluctuations (RMSF) per residue for selected regions of nucleic acids
across multiple molecular dynamics (MD) trajectories. For each trajectory, it identifies the corresponding topology file,
calculates RMSF for atoms in specified residue ranges, and plots the residue-level fluctuations for visual comparison.

Key features:
- Automatically identifies all `.dcd` trajectory files in the current directory.
- Matches each trajectory to a topology file (`.pdb` or `.prmtop`) using longest shared filename prefix.
- Selects atoms within user-defined residue ranges (default: 17–27 and 58–68) and belonging to nucleic acid residues.
- Aligns all trajectory frames to the first frame to remove global motion.
- Computes per-atom RMSF values and averages them per residue.
- Plots and compares RMSF per residue for each trajectory in a single figure.

Required input files:
- One or more MD trajectory files in `.dcd` format.
- Corresponding topology files (`.pdb` or `.prmtop`) for each trajectory, with matching base filenames (e.g., `sample.dcd` and `sample.pdb`).
- The topology must contain nucleic acid residues (A, U, G, C or their DNA equivalents: DA, DT, DG, DC).
- The residue numbering must include the specified ranges (default: 17–27 and 58–68).

Requirements:
- Python with `mdtraj`, `numpy`, `matplotlib`, and `glob` libraries installed.
- All input files should be in the same working directory as the script.

Output:
- A plot titled "RMSF Comparison Across Trajectories" saved as `rmsf_comparison.png`.
- RMSF values plotted per residue for the specified regions in each trajectory, labeled by filename.

This script is ideal for focused RMSF analysis of loop regions, kissing interfaces, or other specific segments of RNA/DNA across different simulations (e.g., varying ionic conditions or mutations).
"""


import mdtraj as md
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def find_best_match(base_name, candidates):
    """Find the topology file with the longest common prefix with the trajectory name."""
    best_match = None
    best_len = 0
    for c in candidates:
        common_len = len(os.path.commonprefix([base_name, c]))
        if common_len > best_len:
            best_len = common_len
            best_match = c
    return best_match

def compute_rmsf_per_residue(traj):
    """Compute RMSF averaged over all atoms in selected residues (nucleic only)."""

    # Define residue range(s)
    selected_ranges = [(17, 27), (58, 68)]

    # Valid nucleotide residue names
    nucleotide_resnames = {'A', 'U', 'G', 'C', 'DA', 'DT', 'DG', 'DC'}

    # Select atoms in specified ranges
    nuc_atoms = [
        atom.index for atom in traj.top.atoms
        if atom.residue.name in nucleotide_resnames and
        any(start <= atom.residue.resSeq <= end for start, end in selected_ranges)
    ]

    if len(nuc_atoms) == 0:
        raise ValueError("No nucleic atoms found in specified residue ranges.")

    # Superpose and calculate RMSF
    traj.superpose(traj[0], atom_indices=nuc_atoms)

    mean_xyz = traj.xyz.mean(axis=0)
    fluctuations = traj.xyz - mean_xyz
    rmsf_per_atom = np.sqrt(np.mean(np.sum(fluctuations**2, axis=2), axis=0))

    # Map atom RMSF to residues
    res_rmsf = {}
    for atom_idx, rmsf_val in zip(range(len(rmsf_per_atom)), rmsf_per_atom):
        atom = traj.top.atom(atom_idx)
        res = atom.residue
        if (res.name in nucleotide_resnames and
            any(start <= res.resSeq <= end for start, end in selected_ranges)):
            res_name = f"{res.name}{res.resSeq}"
            res_rmsf.setdefault(res_name, []).append(rmsf_val)

    # Sort and average
    res_labels = sorted(res_rmsf.keys(), key=lambda r: int(''.join(filter(str.isdigit, r))))
    avg_rmsfs = [np.mean(res_rmsf[r]) for r in res_labels]

    return res_labels, avg_rmsfs


# === Gather all DCD files ===
dcd_files = sorted(glob.glob("*.dcd"))
topo_candidates = glob.glob("*.pdb") + glob.glob("*.prmtop")

# === Plotting setup ===
plt.figure(figsize=(12, 6))

for traj_file in dcd_files:
    base_name = os.path.splitext(traj_file)[0]
    topo_file = find_best_match(base_name, topo_candidates)
    if topo_file is None:
        print(f"❌ No topology found for {traj_file}, skipping.")
        continue

    print(f"📂 Processing: {traj_file}")
    print(f"📎 Topology:   {topo_file}")

    traj = md.load(traj_file, top=topo_file)

    try:
        res_labels, avg_rmsf = compute_rmsf_per_residue(traj)
        plt.plot(res_labels, avg_rmsf, label=base_name, linewidth=1.8)
    except Exception as e:
        print(f"⚠️ Error processing {traj_file}: {e}")
        continue

# === Finalize plot ===
plt.xlabel("Residue")
plt.ylabel("RMSF (nm)")
plt.title("RMSF Comparison Across Trajectories")
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("rmsf_comparison.png", dpi=300)
plt.show()

