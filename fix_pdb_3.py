"""
===============================================================
PDB Modifier: Residue Renumbering and Region-Based Annotation
===============================================================

Purpose:
--------
This script processes a PDB file to perform the following tasks on all 
`ATOM` and `HETATM` records:

1. Renumbering:
   - If the original residue number is ≥ 83, subtract 41 to reassign the
     residue index starting from 42 instead of continuing linearly.
   - This is useful for converting dimeric structures back into a common
     numbering scheme (e.g., [1–41, 42–82] → [1–82]).

2. Occupancy and B-factor Annotation:
   - Occupancy is set to `0.00` for all atoms.
   - B-factor is set to `1.00` only for atoms in:
     - Residues 17–27 (first loop or region of interest)
     - Residues 58–68 (second loop or region of interest)
   - All other residues are assigned a B-factor of `0.00`.

Use Cases:
----------
- Useful for visual emphasis in structural viewers (e.g., PyMOL, VMD) by coloring 
  regions based on B-factor.
- Helps preserve dual-chain residue mapping (e.g., monomer A and B) in simulations.
- Prepares consistent residue numbering across monomers after translation or alignment.

PDB Formatting Notes:
---------------------
- Residue number:    Columns 23–26
- Occupancy:         Columns 55–60
- B-factor:          Columns 61–66

Behavior:
---------
- Non-coordinate lines (e.g., REMARK, TER, END) are preserved unchanged.
- Lines that fail to parse residue numbers are also written unchanged.

Usage:
------
>>> modify_pdb_columns("cs.pdb", "cs_tl.pdb")

Input Requirements:
-------------------
- Standard-format PDB file (`input_pdb`)
- The file should have residue numbers compatible with parsing at columns 23–26.

Output:
-------
✅ Modified PDB written to: cs_tl.pdb

"""

def modify_pdb_columns(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    resid = int(line[22:26])  # original residue number
                    occupancy = " 0.00"
                    bfactor = " 1.00" if 17 <= resid <= 27 else " 0.00"
                    bfactor = " 1.00" if 58 <= resid <= 68 else " 0.00"

                    # Renumber residue: if resid >= 83, subtract 41 to start at 42
                    if resid >= 83:
                        new_resid = resid - 41
                    else:
                        new_resid = resid

                    # Format new residue number to fit into columns 23-26 (4 chars, right-aligned)
                    new_resid_str = f"{new_resid:>4}"

                    # Build new line with updated residue number, occupancy and B-factor
                    newline = (
                        line[:22] +  # up to residue number start
                        new_resid_str +  # new residue number (4 chars)
                        line[26:54] +  # rest of atom line before occupancy/bfactor
                        f"{occupancy:>6}{bfactor:>6}" +  # occupancy and b-factor columns
                        line[66:]  # rest of line
                    )
                    outfile.write(newline)
                except ValueError:
                    # If residue number parse fails, write line unchanged
                    outfile.write(line)
            else:
                outfile.write(line)

    print(f"✅ Modified PDB written to: {output_pdb}")

modify_pdb_columns("cs.pdb", "cs_tl.pdb")

