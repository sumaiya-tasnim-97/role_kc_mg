"""
This script computes and compares the root-mean-square fluctuations (RMSF) per residue for nucleic acid systems
across multiple molecular dynamics (MD) trajectories. For each trajectory, it identifies the corresponding topology file,
calculates residue-level RMSF, and plots the results for visual comparison.

Key steps:
- Automatically identify `.dcd` trajectory files in the working directory.
- Match each trajectory to its topology file (`.pdb` or `.prmtop`) using filename prefix similarity.
- Load each trajectory using MDTraj and select atoms belonging to nucleic acid residues.
- Align all frames to the first frame to remove global motion.
- Calculate per-atom RMSF, then average it over all atoms in each residue to obtain residue-level flexibility.
- Plot residue-wise RMSF values for each trajectory on the same graph for comparison.

Required input files (must be present in the working directory):
- One or more trajectory files in `.dcd` format (e.g., `kc_mg_0_uw.dcd`, `kc_mg_8_uw.dcd`, etc.).
- Corresponding topology files (`.pdb` or `.prmtop`) for each trajectory. The filenames should share a prefix with the
  trajectory files (e.g., `kc_mg_0_uw.dcd` and `kc_mg_0_uw.pdb`).
- Each topology must contain nucleic acid residues (A, U, G, C, or DNA equivalents: DA, DT, DG, DC).

Requirements:
- Python with `mdtraj`, `numpy`, `matplotlib`, and `glob` available.
- Consistent naming conventions between trajectory and topology files to enable automatic matching.
- The trajectory files must contain nucleic acid residues for atom selection to succeed.

Output:
- A plot titled "RMSF Comparison Across Trajectories" saved as `rmsf_comparison.png` in the working directory.
- RMSF values plotted per residue for each input trajectory with clear labels and legend.

This script is useful for comparing local flexibility across simulations of related RNA/DNA systems, such as different ion conditions or sequence variants.
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
    """Compute RMSF averaged over all atoms in each residue (nucleic only)."""
    # Select only atoms from nucleic residues
    nucleotide_resnames = {'A', 'U', 'G', 'C', 'DA', 'DT', 'DG', 'DC'}
    nuc_atoms = [atom.index for atom in traj.top.atoms if atom.residue.name in nucleotide_resnames]

    if len(nuc_atoms) == 0:
        raise ValueError("No nucleic atoms found in topology.")

    traj.superpose(traj[0], atom_indices=nuc_atoms)

    # Per-atom RMSF
    mean_xyz = traj.xyz.mean(axis=0)
    fluctuations = traj.xyz - mean_xyz
    rmsf_per_atom = np.sqrt(np.mean(np.sum(fluctuations**2, axis=2), axis=0))

    # Map atom RMSF to residues
    res_rmsf = {}
    for atom_idx, rmsf_val in zip(range(len(rmsf_per_atom)), rmsf_per_atom):
        atom = traj.top.atom(atom_idx)
        if atom.residue.name in nucleotide_resnames:
            res_id = atom.residue.resSeq
            res_name = f"{atom.residue.name}{res_id}"
            res_rmsf.setdefault(res_name, []).append(rmsf_val)

    # Average RMSF per residue
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

