"""
Universal post-processing for Targeted Molecular Dynamics (TMD) trajectories
to estimate a Potential of Mean Force (PMF) along a user-chosen 1D coordinate
(default: RMSD) and compute multiple binding free-energy (ΔG_bind) estimates.

You can adapt this to any system by editing the CONFIG block below. The core
logic (parsing, work integration, interpolation, Jarzynski averaging,
ΔG estimators, plotting) is unchanged.
"""

# ============================ CONFIG (edit me) ============================

CONFIG = {
    # Files
    "LOG_GLOB": "*.log",          # Glob pattern for TMD logs

    # Reaction coordinate grid (direction preserved; only endpoints change)
    "GRID_START": 12.5,            # coordinate value at index 0
    "GRID_END": 1.5,               # coordinate value at index -1
    "GRID_NPTS": 500,              # number of grid points

    # Temperature & bias
    "TEMPERATURE_K": 310.0,        # Kelvin
    "SPRING_K": 40.0,              # kcal/mol/Å^2

    # PMF region definitions for ΔG (purely geometric; tune to your system)
    "UNBOUND_RANGE": (10.0, 12.0), # tuple defining the unbound window on the grid
    "BOUND_USE_MIN_POINT": True,   # keep bound = argmin coordinate (matches original logic)
    "BOUND_SHADE_WIDTH": 0.02,     # visual shading width (in coordinate units) near min

    # Plateau detection (auto "flat" region finder)
    "PLATEAU_SLOPE_THRESHOLD": 0.1,  # kcal/mol per unit-coordinate

    # Plot cosmetics
    "COORDINATE_LABEL": r"RMSD ($\AA$)",  # axis label for the coordinate
    "TITLE": "PMF and ΔG_bind from TMD",
    "UNBOUND_LABEL": "Putative unbound region",
    "BOUND_LABEL": "Putative bound region",

    # TMD log parsing — use these if your log format differs.
    # By default, the code looks for lines starting with 'TMD' and assumes:
    # parts[1] = step (int), parts[4] = target coordinate, parts[5] = current coordinate
    "LINE_PREFIX": "TMD",
    "STEP_COL": 1,
    "TARGET_COL": 4,
    "CURRENT_COL": 5,
}

# ======================== END CONFIG (logic below) ========================

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def read_tmd_log(logfile, *,
                 line_prefix=CONFIG["LINE_PREFIX"],
                 step_col=CONFIG["STEP_COL"],
                 target_col=CONFIG["TARGET_COL"],
                 current_col=CONFIG["CURRENT_COL"]):
    """
    Parse a TMD log file to extract simulation steps, current coordinate (e.g., RMSD),
    and restraint deviation (target - current).

    Parameters
    ----------
    logfile : str
        Path to a single TMD log file.
    line_prefix : str
        Prefix that marks lines containing TMD fields (default: 'TMD').
    step_col : int
        Zero-based index of the 'step' token in line split.
    target_col : int
        Zero-based index of the 'target coordinate' token.
    current_col : int
        Zero-based index of the 'current coordinate' token.

    Returns
    -------
    steps : np.ndarray
    coords : np.ndarray
    forces : np.ndarray
    """
    steps, coords, forces = [], [], []
    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith(line_prefix):
                try:
                    parts = line.strip().split()
                    step = int(parts[step_col])
                    target_val = float(parts[target_col])
                    current_val = float(parts[current_col])
                    force = target_val - current_val
                    steps.append(step)
                    coords.append(current_val)
                    forces.append(force)
                except Exception as e:
                    print(f"⚠️ Error in {logfile}: {line.strip()}\n{e}")
    return np.array(steps), np.array(coords), np.array(forces)


def compute_work(forces, k, dcoord):
    """
    Integrate the work profile along the coordinate under a harmonic bias.
    Work = -0.5 * k * Σ [ (force)^2 * Δcoord ]
    """
    return -0.5 * k * np.cumsum(forces**2 * dcoord)


def jarzynski_equality(work, kT):
    """
    Apply Jarzynski’s equality to an ensemble of work profiles at each grid point.
    F = -kT * ln <exp(-(W - Wmin)/kT)> + Wmin
    """
    work_min = np.min(work, axis=1, keepdims=True)
    exp_factor = np.exp(-(work - work_min) / kT)
    avg_exp = np.mean(exp_factor, axis=1)
    return -kT * np.log(avg_exp) + work_min.flatten()


def smooth(y, window=11, poly=3):
    """
    Savitzky–Golay smoothing with guards for short arrays / small windows.
    """
    if len(y) < window:
        window = len(y) - (1 - len(y) % 2)
    if window < poly + 2:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)


def bootstrap_dg(pmf, distances, bound_mask, unbound_mask, n_boot=1000, seed=42):
    """
    Bootstrap ΔG between bound and unbound regions by resampling PMF points.
    (Resamples grid points, not trajectories.)
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
    Standard-state (1 M) correction to convert a PMF well depth to ΔG°.
    ΔG° = RT ln(1660)  (with volume in Å^3)
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


def main():
    # === Parameters from CONFIG ===
    k = float(CONFIG["SPRING_K"])
    T = float(CONFIG["TEMPERATURE_K"])
    kB = 0.0019872041
    kT = kB * T
    grid_start = float(CONFIG["GRID_START"])
    grid_end = float(CONFIG["GRID_END"])
    npts = int(CONFIG["GRID_NPTS"])
    unbound_lo, unbound_hi = CONFIG["UNBOUND_RANGE"]
    slope_thresh = float(CONFIG["PLATEAU_SLOPE_THRESHOLD"])

    # Common coordinate grid (direction preserved)
    coord_uniform = np.linspace(grid_start, grid_end, npts)

    # === Collect log files ===
    logfiles = sorted(glob.glob(CONFIG["LOG_GLOB"]))
    if not logfiles:
        print("❌ No log files found.")
        return

    all_work_interp, smooth_work = [], []

    # === Process each trajectory ===
    for logfile in logfiles:
        steps, coords, forces = read_tmd_log(logfile)
        if len(coords) == 0:
            continue
        dcoord = np.gradient(coords)
        work = compute_work(forces, k, dcoord)

        try:
            interp_func = interp1d(coords, work, bounds_error=False, fill_value="extrapolate")
            work_interp = interp_func(coord_uniform)
            all_work_interp.append(work_interp)
            smooth_work.append(smooth(work_interp, window=11, poly=3))
        except Exception as e:
            print(f"⚠️ Interpolation failed for {logfile}: {e}")

    if not all_work_interp:
        print("❌ No usable work data.")
        return

    # Arrays: shape (n_points, n_traj)
    all_work_interp = np.array(all_work_interp).T
    smooth_work = np.array(smooth_work).T

    # === Jarzynski PMFs ===
    pmf_raw = jarzynski_equality(all_work_interp, kT)
    pmf_smooth = smooth(pmf_raw, window=11, poly=3)
    pmf_w_smooth = jarzynski_equality(smooth_work, kT)

    # Simple spread proxy (SEM of work profiles; not a Jarzynski CI)
    pmf_std = np.std(all_work_interp, axis=1) / np.sqrt(all_work_interp.shape[1])

    # === ΔG_bind region masks (universal, configurable) ===
    distances = coord_uniform
    # Bound region: minimum coordinate point (matches original logic)
    bound_mask = distances == np.min(distances) if CONFIG["BOUND_USE_MIN_POINT"] else distances == distances[0]
    # Unbound region: configurable window
    unbound_mask = (distances > unbound_lo) & (distances < unbound_hi)

    # === Plateau detection (auto) ===
    grad = np.abs(np.gradient(pmf_smooth, distances))
    plateau_start_index = np.argmax(grad < slope_thresh)
    dg_auto_plateau = pmf_smooth[plateau_start_index] - np.min(pmf_smooth)

    grad2 = np.abs(np.gradient(pmf_w_smooth, distances))
    plateau_start_index2 = np.argmax(grad2 < slope_thresh)
    dg_auto_plateau_2 = pmf_w_smooth[plateau_start_index2] - np.min(pmf_w_smooth)

    # === ΔG_bind estimators (same logic, reported twice for both PMFs) ===
    dg_pointwise = pmf_smooth[-1] - np.min(pmf_smooth)
    dg_avg = np.mean(pmf_smooth[unbound_mask]) - np.mean(pmf_smooth[bound_mask])
    dg_maxmin = np.max(pmf_smooth) - np.min(pmf_smooth)
    dg_boot_mean, dg_boot_std = bootstrap_dg(pmf_smooth, distances, bound_mask, unbound_mask)
    dg_std_corr = standard_state_correction(T)

    dg_pointwise_2 = pmf_w_smooth[-1] - np.min(pmf_w_smooth)
    dg_avg_2 = np.mean(pmf_w_smooth[unbound_mask]) - np.mean(pmf_w_smooth[bound_mask])
    dg_maxmin_2 = np.max(pmf_w_smooth) - np.min(pmf_w_smooth)
    dg_boot_mean_2, dg_boot_std_2 = bootstrap_dg(pmf_w_smooth, distances, bound_mask, unbound_mask)
    dg_std_corr_2 = standard_state_correction(T)

    # === Identify closest-matching trajectory ===
    closest_file = find_closest_file(pmf_raw, all_work_interp, logfiles)
    print(f"\n🔍 Closest trajectory to PMF (final value): {closest_file}")

    # === Final ΔG output ===
    print("\n===== Estimated Binding Free Energy from PMF =====")
    print(f"1. Pointwise ΔG_bind (plateau - min):       {dg_pointwise:.2f} kcal/mol")
    print(f"2. Region-averaged ΔG_bind:                 {dg_avg:.2f} kcal/mol")
    print(f"3. Max - Min PMF range:                     {dg_maxmin:.2f} kcal/mol")
    print(f"4. Bootstrapped ΔG_bind:                    {dg_boot_mean:.2f} ± {dg_boot_std:.2f} kcal/mol")
    print(f"5. Standard-state correction (1 M):         {dg_std_corr:.2f} kcal/mol")
    print(f"6. ΔG_bind corrected to standard state:     {(dg_avg + dg_std_corr):.2f} kcal/mol")
    print(f"7. Detect plateau automatically:            {dg_auto_plateau:.2f} kcal/mol")

    print("\n===== Estimated Binding Free Energy from PMF (Smoothed Work) =====")
    print(f"1. Pointwise ΔG_bind (plateau - min):       {dg_pointwise_2:.2f} kcal/mol")
    print(f"2. Region-averaged ΔG_bind:                 {dg_avg_2:.2f} kcal/mol")
    print(f"3. Max - Min PMF range:                     {dg_maxmin_2:.2f} kcal/mol")
    print(f"4. Bootstrapped ΔG_bind:                    {dg_boot_mean_2:.2f} ± {dg_boot_std_2:.2f} kcal/mol")
    print(f"5. Standard-state correction (1 M):         {dg_std_corr_2:.2f} kcal/mol")
    print(f"6. ΔG_bind corrected to standard state:     {(dg_avg_2 + dg_std_corr_2):.2f} kcal/mol")
    print(f"7. Detect plateau automatically:            {dg_auto_plateau_2:.2f} kcal/mol")

    # === Plot ===
    plt.figure(figsize=(8, 6))

    # Show smoothed work traces for context
    for i in range(smooth_work.shape[1]):
        plt.plot(distances, smooth_work[:, i], color='gray', alpha=0.2)

    # PMF curves
    plt.plot(distances, pmf_raw, label="Raw PMF", color='orange', linewidth=2)
    plt.plot(distances, pmf_smooth, label="Smoothed PMF", color='blue', linewidth=2)
    plt.plot(distances, pmf_w_smooth, label="Smoothed-Work PMF", color='yellow', linewidth=2)

    # Shaded variability band (SEM of work profiles)
    plt.fill_between(distances, pmf_smooth - pmf_std, pmf_smooth + pmf_std,
                     color='blue', alpha=0.2, label="Std. deviation (work SEM)")

    # Highlight unbound region
    plt.axvspan(unbound_lo, unbound_hi, color='green', alpha=0.1,
                label=CONFIG["UNBOUND_LABEL"])

    # Highlight a thin bound region near the minimum coordinate for visualization
    bound_min = np.min(distances)
    plt.axvspan(bound_min, bound_min + CONFIG["BOUND_SHADE_WIDTH"],
                color='red', alpha=0.1, label=CONFIG["BOUND_LABEL"])

    # Overlay closest trajectory (diagnostic)
    closest_idx = logfiles.index(closest_file)
    plt.plot(distances, all_work_interp[:, closest_idx], color='black',
             linewidth=2, label='Closest trajectory')

    # Direction of coordinate as configured by GRID_START→GRID_END
    if distances[0] > distances[-1]:
        plt.gca().invert_xaxis()

    plt.xlabel(CONFIG["COORDINATE_LABEL"], fontsize=16)
    plt.ylabel("Energy (kcal/mol)", fontsize=16)
    plt.title(CONFIG["TITLE"], fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
