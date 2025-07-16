# =============================================================================
# Hydrogen Bond Distance Tracker (Per-Bond Output) — VMD Tcl Script
#
# Description:
#   This script computes the distance between specified atom pairs that form 
#   hydrogen bonds in a nucleic acid system across all frames of a loaded 
#   molecular dynamics trajectory. For each H-bond, a separate output file is 
#   created containing the frame number and the computed distance (or "NA" if 
#   atoms are missing).
#
# Key Features:
#   - USER-DEFINED BOND LIST: Customize atom pairs and labels for each H-bond.
#   - INDIVIDUAL OUTPUT: Saves one `.dat` file per H-bond for easy plotting.
#   - TRAJECTORY LOOP: Iterates over all frames using VMD's `animate` commands.
#   - SELECTION SAFETY: Handles missing atoms gracefully and prints warnings.
#
# Required Environment:
#   - VMD must be installed and running.
#   - A molecular system (trajectory + topology) must already be loaded in VMD.
#
# Required Inputs:
#   - A trajectory file loaded into VMD with appropriate topology.
#   - H-bonds must be defined via the `hbond_defs` list using the format:
#       {resid1 atom1 resid2 atom2 label}
#     where:
#       - `resid1` and `atom1` define the donor or acceptor atom
#       - `resid2` and `atom2` define the complementary atom
#       - `label` is a user-defined string used as the output filename prefix
#
# Output:
#   - For each defined H-bond, a file named `<label>.dat` is written.
#   - Each file contains two columns: `frame` and `distance` (in Å).
#   - If an atom is missing in a given frame, "NA" is written for that distance.
#
# Example Output File:
#   U21-A63:N3-N1.dat
#   -----------------
#   frame distance
#   0     2.940
#   1     2.957
#   ...
#
# Notes:
#   - Atom selections are based on `resid` and `atom name`—ensure consistency
#     between your input definitions and the actual atom names/resids in VMD.
#   - Output files are saved in the current working directory.
#
# =============================================================================


# === USER INPUT ===

# List of H-bonds to analyze (edit as needed)
set hbond_defs {
    {21 N3 63 N1 U21-A63:N3-N1}
    {21 O4 63 N6 U21-A63:O4-N6}

    {22 N1 62 N3 A22-U62:N1-N3}
    {22 N6 62 O4 A22-U62:N6-O4}

    {23 N3 61 N1 C23-G61:N3-N1}
    {23 O2 61 N2 C23-G61:O2-N2}
    {23 N4 61 O6 C23-G61:N4-O6}

    {20 N1 64 N3 G20-C64:N1-N3}
    {20 N2 64 O2 G20-C64:N2-O2}
    {20 O6 64 N4 G20-C64:O6-N4}
}

# === MAIN LOOP ===

set nframes [molinfo top get numframes]
puts "Number of frames: $nframes"

foreach def $hbond_defs {
    lassign $def resid1 atom1 resid2 atom2 label
    set filename "${label}.dat"
    set outfile [open $filename w]

    puts $outfile "frame distance"
    puts "Processing $label ..."

    for {set frame 0} {$frame < $nframes} {incr frame} {
        animate goto $frame

        set sel1 [atomselect top "resid $resid1 and name $atom1"]
        set sel2 [atomselect top "resid $resid2 and name $atom2"]

        if {([$sel1 num] > 0) && ([$sel2 num] > 0)} {
            set pos1 [lindex [$sel1 get {x y z}] 0]
            set pos2 [lindex [$sel2 get {x y z}] 0]
            set dist [format "%.3f" [veclength [vecsub $pos1 $pos2]]]
        } else {
            set dist "NA"
            puts "Warning: missing atoms for $label at frame $frame"
        }

        puts $outfile "$frame $dist"

        $sel1 delete
        $sel2 delete
    }

    close $outfile
    puts "Saved $filename"
}

puts "✅ All done."

