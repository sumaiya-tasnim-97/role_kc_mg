"""
====================================================================
PDB Occupancy Adjustment Script: Highlighting Specific Residue Ranges
====================================================================

Purpose:
--------
This script modifies the occupancy column (columns 55–60) in a PDB file 
based on residue number. It selectively marks a specific residue range 
(e.g., residues 17–27) with an occupancy value of 1.00, while all others 
are set to 0.00.

This allows structural visualization tools (e.g., PyMOL, VMD, Chimera) 
to highlight or color only the specified regions by occupancy, without 
altering B-factors or spatial coordinates.

Key Features:
-------------
- Retains all original atom line formatting and column alignment.
- Operates only on `ATOM` and `HETATM` records.
- Skips malformed or non-coordinate lines (e.g., `REMARK`, `TER`, `END`).
- Adds padding to lines to ensure proper column formatting for occupancy insertion.

Modification Logic:
-------------------
- Residues 17–27: Occupancy = 1.00
- All others:     Occupancy = 0.00

Input:
------
- `input_pdb`: A standard-format PDB file with residue numbers located at columns 23–26.

Output:
-------
- `output_pdb`: A modified PDB with updated occupancy values.

Usage Example:
--------------
>>> adjust_pdb_occupancy("dimer_separated_along_axis.pdb", "dsaa_mg_one-tl.pdb")

Applications:
-------------
- Region-specific highlighting in molecular graphics.
- Exporting regions of interest for molecular docking or analysis.
- Preprocessing for structure-based alignment or comparison workflows.

Notes:
------
- This script assumes fixed-column formatting typical of PDB standards.
- For additional flexibility (e.g., multiple residue ranges, chain filters), 
  extend the condition block with custom logic or pass ranges as arguments.

"""

def adjust_pdb_occupancy(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    resid = int(line[22:26])

                    # Set occupancy based on residue number
                    if 17 <= resid <= 27:
                        occupancy = 1.00
                    else:
                        occupancy = 0.00
                    occupancy_str = f"{occupancy:6.2f}"

                    # Make sure line is long enough and padded
                    line = line.rstrip('\n').ljust(66)

                    # Rebuild line with updated occupancy
                    new_line = (
                        line[:54] +
                        occupancy_str +      # Occupancy at cols 55–60
                        line[60:] + '\n'
                    )
                    outfile.write(new_line)
                except ValueError:
                    # In case of malformed line, write it as-is
                    outfile.write(line)
            else:
                outfile.write(line)

    print(f"✅ Occupancy-modified PDB written to: {output_pdb}")

# Example usage:
adjust_pdb_occupancy("dimer_separated_along_axis.pdb", "dsaa_mg_one-tl.pdb")

