"""
Batch RMSD Plotting for Multiple RNA Trajectories
--------------------------------------------------

This script processes all `.dcd` trajectory files in the current directory, finds the best-matching
topology file (.prmtop) for each using longest common prefix logic, and computes RMSD values
relative to the first frame. It then plots RMSD vs. time for all trajectories in a single figure.

Features:
- Scans current directory for all `.dcd` files (e.g., from an MD simulation).
- Automatically selects the corresponding `.prmtop` file by prefix similarity.
- Selects nucleic acid atoms using MDTraj's atom selection.
- Aligns each trajectory to its first frame before computing RMSD.
- RMSD values are converted to **angstroms** (Å) for interpretability.
- Time axis is in **microseconds**, assuming 2 fs per frame.
- Produces a single overlayed RMSD vs. time plot for comparison.

Requirements:
- MDTraj
- NumPy
- Matplotlib
- Standard Python libraries: os, glob

How to Use:
1. Place all `.dcd` and corresponding `.prmtop` files in the same directory.
2. Ensure trajectory filenames and topology files share a consistent prefix.
3. Run the script. A plot will appear showing RMSD trends for all detected trajectories.

Example output:
✅ Processing kc_mg_0_uw.dcd
    ↳ Topology:  kc_mg_0.prmtop
✅ Processing kc_mg_8_uw.dcd
    ↳ Topology:  kc_mg_8.prmtop

Plot: RMSD vs Time (in µs), with each trajectory labeled individually.

Note:
- Uncomment the reference file section if you wish to align to an external reference (e.g., a PDB).
"""

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def find_best_match(base_name, candidates):
    # Return candidate with longest matching prefix with base_name
    best_match = None
    best_len = 0
    for c in candidates:
        common_len = len(os.path.commonprefix([base_name, c]))
        if common_len > best_len:
            best_len = common_len
            best_match = c
    return best_match

dcd_files = sorted(glob.glob("*.dcd"))
plt.figure(figsize=(10, 5))

for dcd_file in dcd_files:
    base_name = os.path.splitext(dcd_file)[0]

    prmtop_candidates = glob.glob("*.prmtop")
    prmtop_file = find_best_match(base_name, prmtop_candidates)
    if prmtop_file is None:
        print(f"⚠️ No matching topology found for {dcd_file}")
        continue

#    ref_candidates = glob.glob("*_ref.pdb")
#    ref_file = find_best_match(base_name, ref_candidates)
#    if ref_file is None:
#        print(f"⚠️ No matching reference found for {dcd_file}")
#        continue

    print(f"✅ Processing {dcd_file}")
    print(f"    ↳ Topology:  {prmtop_file}")
#    print(f"    ↳ Reference: {ref_file}")

    # Load files
    traj = md.load(dcd_file, top=prmtop_file)
    reference = traj[0]
# === Select nucleic atoms ===
    nuc_atoms = traj.top.select("nucleic")
    if len(nuc_atoms) == 0:
       raise ValueError("No nucleic atoms found in the topology. Check selection or structure.")


    # Superpose and compute RMSD
    traj.superpose(reference,atom_indices=nuc_atoms)
    rmsd = md.rmsd(traj, reference, atom_indices=nuc_atoms) * 10  # Å
# Your timestep is 2 fs per frame → convert to microseconds (µs)
    time_us = np.arange(len(traj)) * 2e-9  # 2 fs = 2e-9 µs

    label = os.path.splitext(os.path.basename(dcd_file))[0]
    plt.plot(time_us, rmsd, label=label)

plt.xlabel("Time (µs)")
plt.ylabel("RMSD (Å)")
plt.title("RMSD vs Time (Best Prefix Matching)")
plt.legend()
plt.tight_layout()
plt.show()

