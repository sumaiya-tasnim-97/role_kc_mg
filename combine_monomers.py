"""
Merge Two PDB Files While Shifting Atom and Residue Indices of the Second Structure

This script merges two PDB structures into a single file, typically used to construct dimeric
or multimeric RNA complexes from monomer units. It performs the following key operations:

1. **Load and Write First Monomer (Reference) As-Is**:
   - Reads `monomer1_path` and writes its content directly to the output file without modification.

2. **Load Second Monomer and Apply Shifts**:
   - Reads `monomer2_path`, applying:
     - **Atom serial number shift** (`atom_shift`, default = 1324) to avoid overlapping atom indices.
     - **Residue number shift** (`resid_shift`, default = 41) to distinguish monomers in the merged structure.
   - Applies shifts only to lines starting with `ATOM` or `HETATM`, and preserves column formatting.

3. **Output Merged Structure**:
   - The combined structure is saved to `output_path` in standard PDB format.
   - Useful for visualizing or simulating RNA dimers, kissing loops, or symmetric multimers.

### Inputs:
- `monomer1_path`: Path to the original monomer PDB file.
- `monomer2_path`: Path to a second monomer (e.g., transformed version).
- `atom_shift`: Integer offset to apply to atom serial numbers in the second monomer.
- `resid_shift`: Integer offset to apply to residue numbers in the second monomer.

### Output:
- `output_path`: Path to the merged and shifted PDB file.

### Example Use Case:
Used after generating a flipped or rotated copy of an RNA monomer (e.g., `monomer_with_flipped.pdb`)
to produce a cleanly indexed and properly formatted dimer: `combined_shifted.pdb`.


This gives an erronous pdb. run fix_pdb_2.py after this. gets it correct.
"""


def shift_atom_and_resid_and_merge_pdbs(monomer1_path, monomer2_path, output_path, atom_shift=1324, resid_shift=41):
    def parse_and_shift_line(line, atom_offset, resid_offset):
        if line.startswith(("ATOM", "HETATM")):
            try:
                atom_serial = int(line[6:11])
                resnum = int(line[22:26])
                new_atom_serial = atom_serial + atom_offset
                new_resnum = resnum + resid_offset
                # Format new line while preserving alignment
                new_line = (
                    line[:6] + f"{new_atom_serial:5d}" +
                    line[11:22] + f"{new_resnum:4d}" + line[26:]
                )
                return new_line
            except ValueError:
                pass
        return line

    with open(output_path, 'w') as out_file:
        # Write monomer 1 as-is
        with open(monomer1_path, 'r') as f1:
            for line in f1:
                out_file.write(line)

        # Write monomer 2 with atom and resid shifts
        with open(monomer2_path, 'r') as f2:
            for line in f2:
                shifted_line = parse_and_shift_line(line, atom_shift, resid_shift)
                out_file.write(shifted_line)

    print(f"✅ Combined PDB saved to: {output_path}")

shift_atom_and_resid_and_merge_pdbs("cov_monomer.pdb", "monomer_with_flipped.pdb", "combined_shifted.pdb")
