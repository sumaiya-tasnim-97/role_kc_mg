"""
Post-processing script for Targeted Molecular Dynamics (TMD) trajectories
to estimate a Potential of Mean Force (PMF) along RMSD and compute several
binding free energy (ΔG_bind) estimates.

Only user-configurable items:
- Files/log glob
- Log parsing columns (step, target, current)
- Thermo/forcefield (T, k)

All other analysis choices (grid limits, unbound region, axis direction, etc.)
are inferred from the trajectories and the smoothed PMF.

IMPORTANT for this system definition:
- Bound state = highest RMSD (left side)  → single bound point
- Unbound state = low RMSD (right side)   → inferred from PMF plateau
"""

# ===================== USER-CONFIGURABLE (ONLY) =====================

# Files
LOG_GLOB = "*.log"            # Pattern for input logs

# Log parsing
LINE_PREFIX = "TMD"           # Lines to parse start with this
STEP_COL    = 1               # Column index for step (0-based)
TARGET_COL  = 4               # Column index for target RMSD/coord
CURRENT_COL = 5               # Column index for current RMSD/coord

# Thermo/forcefield
TEMP_K    = 310.0             # Temperature (K)
K_B       = 0.0019872041      # Boltzmann constant (kcal/mol/K)
K_SPRING  = 40.0              # Harmonic bias (kcal/mol/Å^2)

# ====================================================================

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def read_tmd_log(logfile):
    """
    Parse a TMD log file to extract simulation steps, current RMSD, and
    restraint deviation (target RMSD - current RMSD).
    """
    steps, rmsds, forces = [], [], []
    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith(LINE_PREFIX):
                try:
                    parts = line.strip().split()
                    step = int(parts[STEP_COL])
                    target_rmsd = float(parts[TARGET_COL])
                    current_rmsd = float(parts[CURRENT_COL])
                    force = target_rmsd - current_rmsd
                    steps.append(step)
                    rmsds.append(current_rmsd)
                    forces.append(force)
                except Exception as e:
                    print(f"⚠️ Error in {logfile}: {line.strip()}\n{e}")
    return np.array(steps), np.array(rmsds), np.array(forces)


def compute_work(forces, k, drmsd):
    """
    Integrate the work profile along RMSD for a trajectory under a harmonic bias.
    Work = -0.5 * k * Σ [ (force)^2 * ΔRMSD ]
    """
    return -0.5 * k * np.cumsum(forces**2 * drmsd)


def jarzynski_equality(work, kT):
    """
    Apply Jarzynski’s equality to an ensemble of work profiles to estimate
    free energy change as a function of RMSD.
    F(R) = -kT * ln ⟨ exp( -[ W_i(R) - Wmin(R) ] / kT ) ⟩ + Wmin(R)
    """
    work_min = np.min(work, axis=1, keepdims=True)
    exp_factor = np.exp(-(work - work_min) / kT)
    avg_exp = np.mean(exp_factor, axis=1)
    return -kT * np.log(avg_exp) + work_min.flatten()


def smooth(y, window=11, poly=3):
    """
    Apply Savitzky–Golay smoothing to a curve.
    """
    if len(y) < window:
        window = len(y) - (1 - len(y) % 2)
    if window < poly + 2:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)


def bootstrap_dg(pmf, distances, bound_mask, unbound_mask, n_boot=1000, seed=42):
    """
    Bootstrap ΔG between bound (single point) and unbound (region) by resampling PMF points.
    (Resamples grid points; not trajectories.)
    """
    rng = np.random.default_rng(seed)
    dg_vals = []
    for _ in range(n_boot):
        pmf_sample = rng.choice(pmf, size=len(pmf), replace=True)
        dg = np.mean(pmf_sample[unbound_mask]) - np.mean(pmf_sample[bound_mask])
        dg_vals.append(dg)
    return np.mean(dg_vals), np.std(dg_vals)


def standard_state_correction(temp_K):
    """
    Standard state correction for 1 M binding free energy from a PMF well depth.
    """
    R = 1.9872041e-3  # kcal/mol·K
    return -R * temp_K * np.log(1 / 1660)


def find_closest_file(pmf, work, filenames):
    """
    Identify the trajectory whose final work value is closest to the PMF endpoint.
    """
    final_pmf_value = pmf[-1]
    work_diff = np.abs(work[-1, :] - final_pmf_value)
    closest_index = np.argmin(work_diff)
    return filenames[closest_index]


def infer_grid_bounds(all_rmsd_lists, npts=500):
    """
    Infer a common coordinate grid from the pooled RMSD across all logs.
    Returns a decreasing grid (max → min) to keep left = high RMSD.
    """
    rmin = min(float(np.min(x)) for x in all_rmsd_lists if len(x) > 0)
    rmax = max(float(np.max(x)) for x in all_rmsd_lists if len(x) > 0)
    # tiny padding to avoid extrapolation issues at boundaries
    pad = 0.01 * (rmax - rmin) if rmax > rmin else 0.0
    start, end = rmax + pad, rmin - pad
    return np.linspace(start, end, npts)


def masks_from_pmf(distances, pmf_smooth, slope_thresh=0.1):
    """
    Build bound/unbound masks from the *smoothed* PMF:

      - Bound (single point): highest RMSD = max(distances).
      - Unbound (region): from the *rightmost* 'flat' index
        (|dF/dR| < slope_thresh) to the right edge (low RMSD side).

    Notes:
    - 'distances' is decreasing (max→min), so the right side has low RMSD.
    - We pick the rightmost flat index to anchor the unbound plateau.
    """
    grad = np.abs(np.gradient(pmf_smooth, distances))
    flat = np.where(grad < slope_thresh)[0]

    if flat.size > 0:
        # rightmost flat index (closest to low RMSD / unbound side)
        plateau_start_index = int(flat[-1])
    else:
        # fallback: take last 15% of grid as 'unbound' if no clear flat region
        plateau_start_index = max(0, int(0.85 * len(distances)))

    # Bound mask = highest RMSD (left edge)
    bound_mask = distances == np.max(distances)

    # Unbound mask: from plateau start to the right edge (low RMSD side)
    unbound_mask = np.zeros_like(distances, dtype=bool)
    unbound_mask[plateau_start_index:] = True

    return bound_mask, unbound_mask, plateau_start_index


def main():
    k = K_SPRING
    T = TEMP_K
    kT = K_B * T

    # -------- PASS 1: read all logs and collect RMSD/force arrays --------
    logfiles_all = sorted(glob.glob(LOG_GLOB))
    if not logfiles_all:
        print("❌ No log files found.")
        return

    parsed = []              # list of (steps, rmsds, forces)
    filenames_used = []      # keep filenames aligned with parsed trajectories

    for logfile in logfiles_all:
        steps, rmsds, forces = read_tmd_log(logfile)
        if len(rmsds) == 0:
            continue
        parsed.append((steps, rmsds, forces))
        filenames_used.append(logfile)

    if not parsed:
        print("❌ No usable data in logs.")
        return

    # Infer grid from all observed RMSDs (max→min so left = high RMSD)
    rmsd_uniform = infer_grid_bounds([r for (_, r, _) in parsed], npts=500)

    # -------- PASS 2: compute work, interpolate, smooth --------
    all_work_interp = []
    smooth_work = []

    for (steps, rmsds, forces) in parsed:
        drmsd = np.gradient(rmsds)
        work = compute_work(forces, k, drmsd)

        try:
            interp_func = interp1d(rmsds, work, bounds_error=False, fill_value="extrapolate")
            work_interp = interp_func(rmsd_uniform)
            all_work_interp.append(work_interp)
            smooth_work.append(smooth(work_interp, window=11, poly=3))
        except Exception as e:
            print(f"⚠️ Interpolation failed for a trajectory: {e}")

    if not all_work_interp:
        print("❌ No usable work data.")
        return

    all_work_interp = np.array(all_work_interp).T   # shape (n_points, n_traj)
    smooth_work = np.array(smooth_work).T

    # -------- Jarzynski PMFs (logic unchanged) --------
    pmf_raw = jarzynski_equality(all_work_interp, kT)
    pmf_smooth = smooth(pmf_raw, window=11, poly=3)
    pmf_w_smooth = jarzynski_equality(smooth_work, kT)

    # variability proxy (SEM of interpolated work across trajectories)
    pmf_std = np.std(all_work_interp, axis=1) / np.sqrt(all_work_interp.shape[1])

    # -------- Masks from smoothed PMF: bound point (max RMSD), unbound region (right) --------
    distances = rmsd_uniform
    bound_mask, unbound_mask, plateau_start_index = masks_from_pmf(
        distances, pmf_smooth, slope_thresh=0.1
    )

    # Also find plateau for the second PMF (from smoothed work)
    grad2 = np.abs(np.gradient(pmf_w_smooth, distances))
    flat2 = np.where(grad2 < 0.1)[0]
    plateau_start_index_2 = int(flat2[-1]) if flat2.size > 0 else max(0, int(0.85 * len(distances)))

    # -------- ΔG estimators (identical formulas; masks now reflect your definition) --------
    dg_auto_plateau = pmf_smooth[plateau_start_index] - np.min(pmf_smooth)
    dg_pointwise    = pmf_smooth[-1] - np.min(pmf_smooth)
    dg_avg          = np.mean(pmf_smooth[unbound_mask]) - np.mean(pmf_smooth[bound_mask])
    dg_maxmin       = np.max(pmf_smooth) - np.min(pmf_smooth)
    dg_boot_mean, dg_boot_std = bootstrap_dg(pmf_smooth, distances, bound_mask, unbound_mask)
    dg_std_corr     = standard_state_correction(T)

    dg_auto_plateau_2 = pmf_w_smooth[plateau_start_index_2] - np.min(pmf_w_smooth)
    dg_pointwise_2    = pmf_w_smooth[-1] - np.min(pmf_w_smooth)
    dg_avg_2          = np.mean(pmf_w_smooth[unbound_mask]) - np.mean(pmf_w_smooth[bound_mask])
    dg_maxmin_2       = np.max(pmf_w_smooth) - np.min(pmf_w_smooth)
    dg_boot_mean_2, dg_boot_std_2 = bootstrap_dg(pmf_w_smooth, distances, bound_mask, unbound_mask)
    dg_std_corr_2     = standard_state_correction(T)

    # -------- Identify closest-matching trajectory (unchanged) --------
    closest_file = find_closest_file(pmf_raw, all_work_interp, filenames_used)
    print(f"\n🔍 Closest trajectory to PMF (final value): {closest_file}")

    # -------- Final ΔG output (unchanged headings & order) --------
    print("\n===== Estimated Binding Free Energy from PMF =====")
    print(f"1. Pointwise ΔG_bind (plateau - min):       {dg_pointwise:.2f} kcal/mol")
    print(f"2. Region-averaged ΔG_bind:                 {dg_avg:.2f} kcal/mol")
    print(f"3. Max - Min PMF range:                     {dg_maxmin:.2f} kcal/mol")
    print(f"4. Bootstrapped ΔG_bind:                    {dg_boot_mean:.2f} ± {dg_boot_std:.2f} kcal/mol")
    print(f"5. Standard-state correction (1 M):         {dg_std_corr:.2f} kcal/mol")
    print(f"6. ΔG_bind corrected to standard state:     {(dg_avg + dg_std_corr):.2f} kcal/mol")
    print(f"7. Detect plateau automatically:            {dg_auto_plateau:.2f} kcal/mol")
    
    print("\n===== Estimated Binding Free Energy from PMF =====")
    print(f"1. Pointwise ΔG_bind (plateau - min):       {dg_pointwise_2:.2f} kcal/mol")
    print(f"2. Region-averaged ΔG_bind:                 {dg_avg_2:.2f} kcal/mol")
    print(f"3. Max - Min PMF range:                     {dg_maxmin_2:.2f} kcal/mol")
    print(f"4. Bootstrapped ΔG_bind:                    {dg_boot_mean_2:.2f} ± {dg_boot_std_2:.2f} kcal/mol")
    print(f"5. Standard-state correction (1 M):         {dg_std_corr_2:.2f} kcal/mol")
    print(f"6. ΔG_bind corrected to standard state:     {(dg_avg_2 + dg_std_corr_2):.2f} kcal/mol")
    print(f"7. Detect plateau automatically:            {dg_auto_plateau_2:.2f} kcal/mol")

    # -------- Plot (bound point at left; unbound shaded at right) --------
    plt.figure(figsize=(8, 6))

    # Smoothed work traces (context)
    for i in range(smooth_work.shape[1]):
        plt.plot(distances, smooth_work[:, i], color='gray', alpha=0.2)

    # PMF curves
    plt.plot(distances, pmf_raw,      label="Raw PMF",          color='orange', linewidth=2)
    plt.plot(distances, pmf_smooth,   label="Smoothed PMF",     color='blue',   linewidth=2)
    plt.plot(distances, pmf_w_smooth, label="Smoothed W PMF",   color='yellow', linewidth=2)

    # Shaded variability band (SEM of work profiles)
    plt.fill_between(distances, pmf_smooth - pmf_std, pmf_smooth + pmf_std,
                     color='blue', alpha=0.2, label="Std. deviation")

    # Unbound shading: from plateau (rightmost flat) to the right edge (low RMSD)
    x_right = distances[-1]
    x_plateau = distances[plateau_start_index]
    plt.axvspan(min(x_plateau, x_right), max(x_plateau, x_right),
                color='green', alpha=0.1, label='Unbound region (Monomers)')

    # Bound point: vertical line at highest RMSD (left side)
    bound_x = np.max(distances)
    plt.axvline(bound_x, color='red', linestyle='--', linewidth=1.5, label='Bound point (Dimer)')

    # Keep visual convention: left = high RMSD, right = low RMSD
    if distances[0] > distances[-1]:
        plt.gca().invert_xaxis()

    plt.xlabel(r"RMSDs ($\AA$)", fontsize=16)
    plt.ylabel("Energy (kcal/mol)", fontsize=16)
    plt.title("PMF and ΔG_bind from TMD", fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
