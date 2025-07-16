"""
This script loads a molecular dynamics trajectory of a nucleic acid system along with its corresponding topology file,
automatically matched by filename. It selects nucleic acid atoms, aligns all frames to the first frame based on these atoms,
and computes the root-mean-square fluctuations (RMSF) per atom to quantify the structural flexibility over the trajectory.

Key steps:
- Automatically find the matching topology (.pdb or .prmtop) file for the given trajectory.
- Load trajectory and topology using MDTraj.
- Select nucleic acid atoms for analysis.
- Superpose (align) all frames to the first frame to remove global motion.
- Calculate per-atom RMSF for the nucleic acid region.
- Report average and standard deviation of RMSF as a measure of local structural fluctuations.

Required input files (must be present in the working directory):
- A trajectory file (e.g., `.dcd`, `.xtc`, or similar) containing the molecular dynamics simulation frames.
- A topology file describing the system structure and atom types, in either `.pdb` or `.prmtop` format.
  The topology filename should share the base name with the trajectory file (e.g., `kc_mg_8_uw.dcd` and `kc_mg_8_uw.pdb`).

Requirements:
- Python with `mdtraj` and `numpy` installed.
- The trajectory must contain nucleic acid residues for the nucleic acid atom selection to succeed.

This script is useful for analyzing flexibility and dynamics of nucleic acid motifs in MD simulations.
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

traj.superpose(reference,atom_indices=nuc_atoms)

# Compute RMSF for those atoms
mean_structure = traj.xyz.mean(axis=0)          # average structure
fluctuations = traj.xyz - mean_structure        # deviations
rmsf = np.sqrt(np.mean(np.sum(fluctuations**2, axis=2), axis=0))  # per atom

# Restrict to nucleic region
region_rmsf = rmsf[nuc_atoms]

# Calculate statistics
avg_rmsf = np.mean(region_rmsf)
std_rmsf = np.std(region_rmsf)

# Report
print(f"📊 Average RMSF (residues 10–20): {avg_rmsf:.3f} nm")
print(f"±   Std Dev:                     {std_rmsf:.3f} nm")

