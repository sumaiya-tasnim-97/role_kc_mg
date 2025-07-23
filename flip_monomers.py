
"""
Generate a Symmetric Dimer by Reflecting an RNA Monomer Across Its Principal Axis

This script uses MDAnalysis to construct a symmetric dimer of an RNA monomer by performing
a geometric reflection and translation of the original structure. It proceeds through the following steps:

1. **Load and Center**:
   - Reads in a monomer PDB file and calculates the geometric center.
   - Recenters all atomic coordinates around the origin for easier manipulation.

2. **Compute Principal Axis**:
   - Uses the inertia tensor to identify the longest principal axis of the monomer.
   - This axis defines the reflection plane normal.

3. **Reflect and Translate**:
   - Constructs a reflection matrix to flip the molecule across the principal axis.
   - Applies a translation along the same axis (e.g., +80 Å) to avoid overlap with the original.

4. **Chain and Residue Assignment**:
   - Assigns the new reflected monomer to chain ID 'B'.
   - Renumbers its residues starting from 42 to distinguish from the original monomer.

5. **Combine and Write Output**:
   - Merges the original and flipped monomers into a single AtomGroup.
   - Outputs the new dimer structure as a PDB file.

This approach is useful for modeling RNA kissing complexes or other symmetric dimeric assemblies
based on a single monomer unit. The resulting file `monomer_with_flipped.pdb` can be directly
used for visualization, further simulation, or structure-based analysis.

Input:  `cov_monomer.pdb` (single monomer)
Output: `monomer_with_flipped.pdb` (centered dimer with mirrored monomer)
"""
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.lib.transformations import rotation_matrix as rotmat

# Load the original PDB
u = mda.Universe("cov_monomer.pdb")
atoms = u.atoms

# Get coordinates and center
coords = atoms.positions.copy()
center = coords.mean(axis=0)

# Center the coordinates
centered_coords = coords - center

# Compute principal axes from the inertia tensor
I = np.dot(centered_coords.T, centered_coords)
eigvals, eigvecs = np.linalg.eigh(I)
principal_axis = eigvecs[:, np.argmax(eigvals)]  # Longest principal axis

# Create a reflection matrix across the principal axis
# Flip all coordinates along this axis (mirror)
reflection = np.identity(3)
reflection -= 2 * np.outer(principal_axis, principal_axis)

# Apply reflection and shift
flipped_coords = np.dot(centered_coords, reflection.T) + center + principal_axis * 80  # Shift away

# Create new atomgroup with flipped coordinates
flipped = atoms.copy()
flipped.positions = flipped_coords

# Set chain ID to B
flipped.segments.segids = ['B']

# Optional: renumber residues starting at 42 (assuming original monomer ends at 41)
flipped.residues.resids = np.arange(42, 42 + len(flipped.residues))

# Combine original and flipped coordinates
combined_coords = np.vstack([atoms.positions, flipped.positions])
combined_atoms = atoms.concatenate(flipped)

# Write to PDB
with mda.coordinates.PDB.PDBWriter("monomer_with_flipped.pdb", multiframe=False) as pdb:
    pdb.write(combined_atoms)

print("✅ Saved as 'monomer_with_flipped.pdb'")
