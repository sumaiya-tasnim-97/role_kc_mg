# ==============================================================
# VMD Script: Separate RNA Monomers Along Principal Axis
# ==============================================================
# Purpose:
# This script separates two RNA monomers (e.g., kissing dimers) along
# the vector connecting their centers of mass, enabling visualization 
# or further computation on unstacked/individualized monomers.
#
# Key Steps:
# 1. Goes to the last frame of the currently loaded trajectory.
# 2. Defines two monomers based on residue ranges and labels them 
#    with distinct chain identifiers ("A" and "B").
# 3. Selects all Mg²⁺ or Na⁺ ions within 6 Å of each chain (proximal ions).
# 4. Groups each monomer with its respective nearby ions.
# 5. Computes the center of mass (COM) of each monomer.
# 6. Constructs and normalizes the vector from monomer A to monomer B.
# 7. Translates chain B and its associated ions along this vector by 
#    a user-defined distance (default: 15 Å).
# 8. Writes the transformed structure as a new PDB file.
#
# Output:
# - `dimer_separated_along_axis.pdb`: PDB structure with monomers
#    separated along their COM–COM axis by 15 Å.
#
# Requirements:
# - This script assumes the topology and trajectory are already loaded
#   into VMD, and that residues 1–41 and 42–82 correspond to monomers A and B.
# - The script uses frame-based selection for accurate atom positioning.
# - It operates on the *last frame* of the trajectory.
#
# Customizable Parameters:
# - `scale`: Change this value to adjust the separation distance.
#
# Usage:
# Load your structure and trajectory in VMD, then run:
# ```tcl
# source separate_monomers.tcl
# ```

# === Go to the last frame ===
set numframes [molinfo top get numframes]
set lastframe [expr {$numframes - 1}]
animate goto $lastframe

# === Label chains ===
set monA [atomselect top "resid 1 to 41" frame $lastframe]
$monA set chain "A"
set monB [atomselect top "resid 42 to 82" frame $lastframe]
$monB set chain "B"

# === Reselect by chain ID ===
set monA [atomselect top "chain A" frame $lastframe]
set monB [atomselect top "chain B" frame $lastframe]
set ionsA [atomselect top "resname MG NA and within 6 of chain A" frame $lastframe]
set ionsB [atomselect top "resname MG NA and within 6 of chain B" frame $lastframe]

# === Group selections ===
set groupA [atomselect top "(chain A) or (resname MG NA and within 6 of chain A)" frame $lastframe]
set groupB [atomselect top "(chain B) or (resname MG NA and within 6 of chain B)" frame $lastframe]

# === Compute center of mass for each monomer ===
set comA [measure center $monA weight mass]
set comB [measure center $monB weight mass]

# === Compute the direction vector from A to B ===
set dirVec [vecsub $comB $comA]

# === Normalize the direction vector ===
set dirNorm [vecnorm $dirVec]

# === Scale by desired translation distance (e.g., 15 Å) ===
set scale 15.0
set moveVec [vecscale $scale $dirNorm]

# === Move chain B and its ions along the vector ===
$groupB moveby $moveVec

# === Save output PDB ===
set all [atomselect top all frame $lastframe]
$all writepdb dimer_separated_along_axis.pdb

# === Clean up ===
foreach sel [list $monA $monB $ionsA $ionsB $groupA $groupB $all] {
    $sel delete
}
