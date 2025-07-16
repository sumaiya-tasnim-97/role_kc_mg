"""
RNA Trajectory RMSD Analysis Script
-----------------------------------

This script computes the root-mean-square deviation (RMSD) of an RNA molecular dynamics (MD) trajectory 
relative to its first frame, focusing on nucleic acid atoms.

Key Features:
- Automatically selects the best-matching topology file (.pdb or .prmtop) for the given trajectory.
- Loads the trajectory using MDTraj.
- Selects nucleic atoms using MDTraj's atom selection language.
- Aligns all trajectory frames to the first (reference) frame using nucleic atoms.
- Computes the per-frame RMSD with respect to the reference structure.
- Reports the average and standard deviation of the RMSD values.

How to Use:
1. Change the value of `traj_file` to point to your trajectory (.dcd or .xtc).
2. Ensure the corresponding topology (.pdb or .prmtop) is in the same directory.
3. Run the script to get RMSD statistics and confirm structural stability or drift.

Dependencies:
- MDTraj
- NumPy
- Standard Python libraries: os, glob

Example output:
✅ Loading trajectory: kc_mg_8_uw.dcd
🔍 Using topology:     kc_mg_8.prmtop

📊 Average RMSD: 0.237 nm
±   Std Dev:     0.012 nm
"""

import mdtraj as md
import numpy as np
import glob
import os

def find_best_match(base_name, candidates):
    best_match = None
    best_len = 0
    for c in candidates:
        common_len = len(os.path.commonprefix([base_name, c]))
        if common_len > best_len:
            best_len = common_len
            best_match = c
    return best_match

# === Specify your trajectory file here ===
traj_file = "kc_mg_8_uw.dcd"  # <- Change to your actual .xtc or .dcd filename

base_name = os.path.splitext(traj_file)[0]

# === Find best-matching topology file (.pdb or .prmtop) ===
topo_candidates = glob.glob("*.pdb") + glob.glob("*.prmtop")
topology_file = find_best_match(base_name, topo_candidates)

if topology_file is None:
    raise FileNotFoundError(f"❌ No matching topology found for: {traj_file}")

print(f"✅ Loading trajectory: {traj_file}")
print(f"🔍 Using topology:     {topology_file}")

# === Load the trajectory ===
traj = md.load(traj_file, top=topology_file)

# Use first frame as reference
reference = traj[0]

# === Select nucleic atoms ===
nuc_atoms = traj.top.select("nucleic")
#nuc_atoms = traj.top.select("resid 17 to 27 or resid 58 to 68")
if len(nuc_atoms) == 0:
     raise ValueError("No nucleic atoms found in the topology. Check selection or structure.")


    # Superpose and compute RMSD
traj.superpose(reference,atom_indices=nuc_atoms)
rmsd = md.rmsd(traj, reference, atom_indices=nuc_atoms)

# Compute average and standard deviation
avg_rmsd = np.mean(rmsd)
std_rmsd = np.std(rmsd)

print(f"\n📊 Average RMSD: {avg_rmsd:.3f} nm")
print(f"±   Std Dev:     {std_rmsd:.3f} nm")

