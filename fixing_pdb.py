"""
===============================================================
PDB Column Modifier: Occupancy & B-Factor Annotation
===============================================================

Purpose:
--------
This script processes a PDB file and modifies the occupancy and B-factor 
fields for all `ATOM` and `HETATM` records. It can be used to highlight 
specific residues (e.g., for visualization in PyMOL or VMD) based on residue 
number criteria.

Modifications:
--------------
- Sets occupancy to 0.00 for all atoms.
- Sets B-factor to:
    - 1.00 for atoms in residues 42 to 82
    - 0.00 for all other residues

This pattern is useful for:
- Coloring or selecting subsets of residues using B-factor heatmaps.
- Differentiating regions (e.g., monomers, loops, interfaces) in structural
  visualization tools.

How it works:
-------------
- Parses each line in the input PDB.
- Identifies ATOM/HETATM records.
- Extracts the residue number (columns 23–26).
- Applies rules to modify occupancy (columns 55–60) and B-factor (columns 61–66).
- Writes the modified line to a new output file.

Usage Example:
--------------
>>> modify_pdb_columns("combined_shifted.pdb", "cs.pdb")

Output:
-------
✅ Modified PDB written to: cs.pdb

Note:
-----
- This script assumes standard PDB formatting; malformed lines may be skipped.
- Non-coordinate records (e.g., REMARK, TER, END) are written unchanged.

"""

def modify_pdb_columns(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    resid = int(line[22:26])  # residue number
                    occupancy = " 0.00"
                    bfactor = " 1.00" if 42 <= resid <= 82 else " 0.00"

                    # Create new line with updated occupancy and b-factor
                    newline = line[:54] + f"{occupancy:>6}{bfactor:>6}" + line[66:]
                    outfile.write(newline)
                except ValueError:
                    # If conversion fails, write line unchanged
                    outfile.write(line)
            else:
                outfile.write(line)

    print(f"✅ Modified PDB written to: {output_pdb}")

# Example usage
modify_pdb_columns("combined_shifted.pdb", "cs.pdb")

