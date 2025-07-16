"""
Mg²⁺–RNA Proximity and Neighborhood Clustering Script (MDAnalysis)

Description:
    This script analyzes the spatial relationship between Mg²⁺ ions and RNA residues 
    in a molecular system using MDAnalysis. It identifies RNA residues within a specified
    cutoff distance of each Mg²⁺ ion (based on the first frame of the trajectory) and then 
    clusters Mg²⁺ ions into groups if they:
      - Share any neighboring RNA residues
      - Or are within a user-defined merge cutoff distance from each other

Purpose:
    Useful for identifying functionally or structurally relevant Mg²⁺ binding regions, 
    such as stabilizing loops, junctions, or kissing interfaces in RNA.

Workflow:
    - Load trajectory and topology using MDAnalysis.
    - Automatically guess atomic elements (for proper ion recognition).
    - Select all Mg²⁺ ions and canonical RNA residues (A, U, G, C).
    - For each Mg²⁺ ion, identify RNA residues within a specified cutoff distance.
    - Group Mg²⁺ ions that:
        • Share neighboring RNA residues
        • Or are closer than `merge_cutoff` Å to each other
    - Output a list of Mg²⁺ groups and the corresponding nearby RNA residues.

User Parameters:
    - `structure`: Path to the structure file (e.g., PDB).
    - `trajectory`: Path to the trajectory file (e.g., DCD).
    - `cutoff`: Distance threshold (in Å) to define Mg²⁺–RNA proximity.
    - `merge_cutoff`: Distance threshold (in Å) for merging Mg²⁺ ions into the same group.

Requirements:
    - Python with MDAnalysis installed.
    - Structure and trajectory files loaded in a compatible format (e.g., `.pdb` and `.dcd`).
    - Assumes that Mg²⁺ ions are labeled as `resname MG` and RNA residues as `A`, `U`, `G`, `C`.

Output:
    - Printed summary of Mg²⁺ clusters (groups) and the set of nearby RNA residues for each.
    - Each group includes a list of Mg²⁺ atom indices and formatted RNA residue names (e.g., G42, A63).

Example Output:
    Group 1: Mg indices [24, 26], Residues: G42, A63, U64
    Group 2: Mg indices [31], Residues: C30, G31

Notes:
    - Only the first frame of the trajectory is used for this spatial analysis.
    - You can adjust residue selection or distance cutoffs for customized use cases (e.g., for DNA, or for tighter binding detection).
"""


import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from collections import defaultdict

# === USER PARAMETERS ===
structure = "cov2_kc_2.pdb"
trajectory = "cov2_2_unwrap.dcd"
cutoff = 5.0              # Å - distance from Mg²⁺ to RNA
merge_cutoff = 5.0        # Å - merge Mg²⁺ if closer than this distance

# === LOAD SYSTEM ===
u = mda.Universe(structure, trajectory)
u.trajectory[0]  # Only using first frame for neighborhood mapping

# === SELECT ATOMS ===
u.guess_TopologyAttrs(context='default', to_guess=['elements'])

# Select Mg²⁺ ions
mg_sel = u.select_atoms("resname MG")

# Select RNA residues explicitly
rna_sel = u.select_atoms("resname A U G C")

print(f"\nFound {len(mg_sel)} Mg²⁺ ions.")
print(f"Found {len(rna_sel.residues)} RNA residues.\n")

# === MAP Mg INDEX TO NEARBY RNA RESIDUES ===
mg_to_residues = {}
mg_coords = mg_sel.positions

for i, mg in enumerate(mg_sel):
    mg_pos = mg.position.reshape(1, 3)
    rna_pos = rna_sel.positions

    dists = distance_array(mg_pos, rna_pos)[0]
    close_indices = [i for i, d in enumerate(dists) if d <= cutoff]

    residues = rna_sel[close_indices].residues
    resid_pairs = set((res.resid, res.resname) for res in residues)

    mg_to_residues[mg.index] = resid_pairs

# === GROUP Mg IONS THAT SHARE NEIGHBORHOODS OR ARE CLOSE ===
visited = set()
groups = []

for i, idx_i in enumerate(mg_to_residues):
    if idx_i in visited:
        continue

    group = {idx_i}
    queue = [idx_i]
    visited.add(idx_i)

    while queue:
        current = queue.pop()
        for j, idx_j in enumerate(mg_to_residues):
            if idx_j in visited:
                continue

            # Check shared residues
            same_residues = not mg_to_residues[current].isdisjoint(mg_to_residues[idx_j])
            # Check Mg-Mg distance
            dist = ((mg_coords[i] - mg_coords[j])**2).sum()**0.5

            if same_residues or dist < merge_cutoff:
                group.add(idx_j)
                queue.append(idx_j)
                visited.add(idx_j)

    groups.append(group)

# === OUTPUT GROUPED REGIONS ===
print("Grouped Mg²⁺ regions and nearby RNA residues:\n")
for i, group in enumerate(groups, 1):
    all_residues = set()
    for mg_idx in group:
        all_residues.update(mg_to_residues[mg_idx])

    if all_residues:
        sorted_residues = sorted(all_residues, key=lambda x: x[0])  # sort by resid
        res_str = ", ".join([f"{r[1]}{r[0]}" for r in sorted_residues])
    else:
        res_str = "[]"

    print(f"Group {i}: Mg indices {sorted(group)}, Residues: {res_str}")

