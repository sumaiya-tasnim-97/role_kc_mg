# =============================================================================
# Hbond Distance Calculation Script for VMD
#
# Description:
#   This script computes hydrogen bond distances for specified atom pairs over
#   all frames of a molecular dynamics trajectory loaded in VMD. For each frame,
#   it selects user-defined residue and atom pairs, calculates their Euclidean
#   distance (if both selections are found), and writes the results to a CSV file.
#
# Key Features:
#   - USER INPUT: Customize the output filename and define hydrogen bond pairs.
#   - CSV OUTPUT: Automatically generate a header line and append distances for
#     each frame in the trajectory.
#   - FRAME LOOP: Iterates through all frames using VMD's 'animate' commands.
#   - Error Handling: Outputs warnings for missing selections in specific frames.
#
# Requirements:
#   - VMD must be installed and the script is intended to run in its Tcl console.
#   - A loaded molecular dynamics trajectory with a valid topology in VMD.
#
# Input Files / Environment:
#   - The script assumes a molecular system is already loaded in VMD (via 'molinfo top').
#   - The trajectory (with multiple frames) and associated topology must be available.
#
# Output:
#   - A CSV file (default: "hbond_distances.csv") created in the current directory,
#     which lists the frame number and the computed distances for each specified hydrogen bond.
#
# Usage:
#   - Modify the 'hbond_defs' list to set the residue numbers, atom names, and labels
#     for the hydrogen bonds of interest.
#   - Set the desired name for the output CSV file in the 'output_file' variable.
#   - Run the script in VMD after loading your molecular system and trajectory.
#
# =============================================================================


# === USER INPUT ===
set output_file "hbond_distances.csv"

set hbond_defs {
    {20 N1 64 N3 G20-C64:N1-N3}
    {20 N2 64 O2 G20-C64:N2-O2}
    {20 O6 64 N4 G20-C64:O6-N4}

    {21 N3 63 N1 U21-A63:N3-N1}
    {21 O4 63 N6 U21-A63:O4-N6}

    {22 N1 62 N3 A22-U62:N1-N3}
    {22 N6 62 O4 A22-U62:N6-O4}

    {23 N3 61 N1 C23-G61:N3-N1}
    {23 O2 61 N2 C23-G61:O2-N2}
    {23 N4 61 O6 C23-G61:N4-O6}
}

# === CSV OUTPUT SETUP ===
set outfile [open $output_file w]
set header "frame"
foreach def $hbond_defs {
    lassign $def resid1 atom1 resid2 atom2 label
    append header ",$label"
}
puts $outfile $header

# === MAIN LOOP ===
set nframes [molinfo top get numframes]
puts "📦 Number of frames detected: $nframes"

for {set frame 0} {$frame < $nframes} {incr frame} {
    animate goto $frame
    set line "$frame"

    foreach def $hbond_defs {
        lassign $def resid1 atom1 resid2 atom2 label

        set sel1 [atomselect top "resid $resid1 and name $atom1"]
        set sel2 [atomselect top "resid $resid2 and name $atom2"]

        if {([$sel1 num] > 0) && ([$sel2 num] > 0)} {
            set pos1 [lindex [$sel1 get {x y z}] 0]
            set pos2 [lindex [$sel2 get {x y z}] 0]
            set dist [format "%.3f" [veclength [vecsub $pos1 $pos2]]]
        } else {
            set dist "NA"
            puts "⚠️  Frame $frame: Missing $label"
        }

        append line ",$dist"
        $sel1 delete
        $sel2 delete
    }

    puts $outfile $line
}

close $outfile
puts "✅ Done. Output written to '$output_file'"
