"""
Post-processing script for Targeted Molecular Dynamics (TMD) trajectories
to estimate a Potential of Mean Force (PMF) along RMSD and compute several
binding free energy (ΔG_bind) estimates.

Bound state definition in this version:
- Bound state = highest RMSD point (left side of plot), which corresponds
  to the kissing dimer (max deviation from the separated monomers target).
- Unbound region is auto-detected from the PMF plateau on the right side.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ----------------------------
# Parsing and core calculations
# ----------------------------

def read_tmd_log(logfile):
    steps, rmsds, forces = [], [], []
    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith("TMD"):
                try:
                    parts = line.strip().split()
                    step = int(parts[1])
                    target_rmsd = float(parts[4])
                    current_rmsd = float(parts[5])
                    force = target_rmsd - current_rmsd
                    steps.append(step)
                    rmsds.append(current_rmsd)
                    forces.append(force)
                except Exception as e:
                    print(f"⚠️ Error in {logfile}: {line.strip()}\n{e}")
    return np.array(steps), np.array(rmsds), np.array(forces)

def compute_work(forces, k, drmsd):
    return -0.5 * k * np.cumsum(forces**2 * drmsd)

def jarzynski_equality(work, kT):
    work_min = np.min(work, axis=1, keepdims=True)
    exp_factor = np.exp(-(work - work_min) / kT)
    avg_exp = np.mean(exp_factor, axis=1)
    return -kT * np.log(avg_exp) + work_min.flatten()

def smooth(y, window=11, poly=3):
    if len(y) < window:
        window = len(y) - (1 - len(y) % 2)
    if window < poly + 2:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)

def bootstrap_dg(pmf, distances, bound_mask, unbound_mask, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    dg_vals = []
    for _ in range(n_boot):
        pmf_sample = rng.choice(pmf, size=len(pmf), replace=True)
        dg = np.mean(pmf_sample[unbound_mask]) - np.mean(pmf_sample[bound_mask])
        dg_vals.append(dg)
    return np.mean(dg_vals), np.std(dg_vals)

def standard_state_correction(temp_K):
    R = 1.9872041e-3  # kcal/mol·K
    return -R * temp_K * np.log(1 / 1660)

def find_closest_file(pmf, work, filenames):
    final_pmf_value = pmf[-1]
    work_diff = np.abs(work[-1, :] - final_pmf_value)
    closest_index = np.argmin(work_diff)
    return filenames[closest_index]

# ----------------------------
# Mask determination (bound/unbound)
# ----------------------------

def masks_from_pmf(distances, pmf_smooth, slope_thresh=0.1):
    """
    Bound = highest RMSD point (max distance).
    Unbound = region from plateau start to smallest RMSD (right side).
    """
    grad = np.abs(np.gradient(pmf_smooth, distances))
    flat = np.where(grad < slope_thresh)[0]
    if flat.size > 0:
        plateau_start_index = int(flat[0])
    else:
        plateau_start_index = max(0, int(0.85 * len(distances)))

    bound_mask = distances == np.max(distances)
    unbound_mask = np.zeros_like(distances, dtype=bool)
    unbound_mask[plateau_start_index:] = True

    return bound_mask, unbound_mask, plateau_start_index

# ----------------------------
# Main analysis pipeline
# ----------------------------

def main():
    k = 40.0  # harmonic spring constant
    T = 310   # Kelvin
    kB = 0.0019872041
    kT = kB * T
    rmsd_uniform = np.linspace(12.5, 1.5, 500)  # decreasing grid

    logfiles = sorted(glob.glob("*.log"))
    if not logfiles:
        print("❌ No log files found.")
        return

    all_work_interp = []
    smooth_work = []

    # Read and process each log
    for logfile in logfiles:
        steps, rmsds, forces = read_tmd_log(logfile)
        if len(rmsds) == 0:
            continue
        drmsd = np.gradient(rmsds)
        work = compute_work(forces, k, drmsd)

        try:
            interp_func = interp1d(rmsds, work, bounds_error=False, fill_value="extrapolate")
            work_interp = interp_func(rmsd_uniform)
            all_work_interp.append(work_interp)
            smooth_work.append(smooth(work_interp))
        except Exception as e:
            print(f"⚠️ Interpolation failed for {logfile}: {e}")

    if not all_work_interp:
        print("❌ No usable work data.")
        return

    all_work_interp = np.array(all_work_interp).T
    smooth_work = np.array(smooth_work).T

    # PMF calculations
    pmf_raw = jarzynski_equality(all_work_interp, kT)
    pmf_smooth = smooth(pmf_raw)
    pmf_w_smooth = jarzynski_equality(smooth_work, kT)
    pmf_std = np.std(all_work_interp, axis=1) / np.sqrt(all_work_interp.shape[1])

    # Masks and ΔG calc
    distances = rmsd_uniform
    bound_mask, unbound_mask, plateau_start_index = masks_from_pmf(distances, pmf_smooth)

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

    # Closest trajectory
    closest_file = find_closest_file(pmf_raw, all_work_interp, logfiles)
    print(f"\n🔍 Closest trajectory to PMF (final value): {closest_file}")

    # Output ΔG results
    print("\n===== Estimated Binding Free Energy from PMF =====")
    print(f"1. Pointwise ΔG_bind:                       {dg_pointwise:.2f} kcal/mol")
    print(f"2. Region-averaged ΔG_bind:                 {dg_avg:.2f} kcal/mol")
    print(f"3. Max - Min PMF range:                     {dg_maxmin:.2f} kcal/mol")
    print(f"4. Bootstrapped ΔG_bind:                    {dg_boot_mean:.2f} ± {dg_boot_std:.2f} kcal/mol")
    print(f"5. Standard-state correction (1 M):         {dg_std_corr:.2f} kcal/mol")
    print(f"6. Corrected to standard state:             {(dg_avg + dg_std_corr):.2f} kcal/mol")
    
    print("\n===== From PMF using smoothed work =====")
    print(f"1. Pointwise ΔG_bind:                       {dg_pointwise_2:.2f} kcal/mol")
    print(f"2. Region-averaged ΔG_bind:                 {dg_avg_2:.2f} kcal/mol")
    print(f"3. Max - Min PMF range:                     {dg_maxmin_2:.2f} kcal/mol")
    print(f"4. Bootstrapped ΔG_bind:                    {dg_boot_mean_2:.2f} ± {dg_boot_std_2:.2f} kcal/mol")
    print(f"5. Standard-state correction (1 M):         {dg_std_corr_2:.2f} kcal/mol")
    print(f"6. Corrected to standard state:             {(dg_avg_2 + dg_std_corr_2):.2f} kcal/mol")

    # Plot
    plt.figure(figsize=(8, 6))
    for i in range(smooth_work.shape[1]):
        plt.plot(distances, smooth_work[:, i], color='gray', alpha=0.2)
    plt.plot(distances, pmf_raw, label="Raw PMF", color='orange', linewidth=2)
    plt.plot(distances, pmf_smooth, label="Smoothed PMF", color='blue', linewidth=2)
    plt.plot(distances, pmf_w_smooth, label="Smoothed W PMF", color='yellow', linewidth=2)
    plt.fill_between(distances, pmf_smooth - pmf_std, pmf_smooth + pmf_std,
                     color='blue', alpha=0.2, label="Std. deviation")

    # Unbound region shading
    unbound_xmin = distances[plateau_start_index]
    plt.axvspan(unbound_xmin, np.min(distances), color='green', alpha=0.1, label='Unbound region')

    # Bound point marker
    bound_x = np.max(distances)
    plt.axvline(bound_x, color='red', linestyle='--', linewidth=1.5, label='Bound point')

    # Closest trajectory
    closest_idx = logfiles.index(closest_file)
    plt.plot(distances, all_work_interp[:, closest_idx], color='black',
             linewidth=2, label='Closest trajectory')

    plt.gca().invert_xaxis()
    plt.xlabel(r"RMSDs ($\AA$)", fontsize=16)
    plt.ylabel("Energy (kcal/mol)", fontsize=16)
    plt.title("PMF and ΔG_bind from TMD", fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
