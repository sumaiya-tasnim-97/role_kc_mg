"""
Post-processing script for Targeted Molecular Dynamics (TMD) trajectories
to estimate a Potential of Mean Force (PMF) along RMSD and compute binding
free energy (ΔG_bind) with data-driven region detection.

- RMSD grid is derived from the data (no hard-coded ranges).
- Bound state = single point from the first TMD steps (robust median),
  drawn as a vertical line.
- Unbound state = FIRST force-based plateau (first low-|dW/dR| region moving
  from high→low RMSD), detected from the median work slope across trajectories.
- ΔG = (mean PMF over this plateau) − (PMF at bound point).  (Sign unchanged)
- Bootstrap resamples WITHIN the plateau (bound is a single point).
- Representative trajectory chosen by plateau mean similarity.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ========= I/O & primitives =========

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
                    force = target_rmsd - current_rmsd  # deviation (Å)
                    steps.append(step)
                    rmsds.append(current_rmsd)
                    forces.append(force)
                except Exception as e:
                    print(f"⚠️ Error in {logfile}: {line.strip()}\n{e}")
    return np.array(steps), np.array(rmsds), np.array(forces)

def compute_work(forces, k, drmsd):
    # cumulative "-1/2 k (ΔR)^2 dR" in energy units
    return -0.5 * k * np.cumsum(forces**2 * drmsd)

def savgol_safe(y, window=11, poly=3):
    if len(y) < 3:
        return y
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
    if window < poly + 2:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)

# ========= Jarzynski (NaN-aware) =========

def jarzynski_equality_nanaware(work_desc, kT):
    """
    work_desc: shape (n_points, n_traj), descending RMSD order.
    Handles NaNs per grid point by dropping NaN trajectories.
    """
    n_points, _ = work_desc.shape
    pmf = np.full(n_points, np.nan)
    for i in range(n_points):
        wi = work_desc[i, :]
        msk = ~np.isnan(wi)
        if not np.any(msk):
            continue
        w = wi[msk]
        wmin = np.min(w)
        avg_exp = np.mean(np.exp(-(w - wmin) / kT))
        pmf[i] = -kT * np.log(avg_exp) + wmin
    return pmf

# ========= FIRST force-based plateau detection =========
# (median |dW/dR| across trajectories; first contiguous low-slope window)

def detect_first_force_plateau(all_work_desc, distances_desc, min_window_frac=0.05):
    """
    Detect the FIRST plateau by looking at the median work slope across trajectories.
    Inputs:
      - all_work_desc: (n_points, n_traj), cumulative work, DESCENDING RMSD order
      - distances_desc: (n_points,), descending RMSD
    Returns:
      start_idx, end_idx, lo_rmsd, hi_rmsd, tau_slope, slope_med_sm
    """
    n, m = all_work_desc.shape
    if n < 10:
        slope_med_sm = np.full(n, np.nan)
        return 0, n-1, distances_desc[0], distances_desc[-1], np.nan, slope_med_sm

    # Per-trajectory slope |dW/dR|
    slopes = np.empty_like(all_work_desc)
    for j in range(m):
        wj = all_work_desc[:, j]
        if np.all(np.isnan(wj)):
            slopes[:, j] = np.nan
        else:
            slopes[:, j] = np.abs(np.gradient(wj, distances_desc))

    # Median across trajectories (robust) and smooth
    slope_med = np.nanmedian(slopes, axis=1)
    window = max(11, (n // 20) * 2 + 1)
    slope_med_sm = savgol_safe(slope_med, window=window, poly=3)

    # Robust low-slope threshold τ from the lowest 30% of values
    k_low = max(int(0.30 * n), 8)
    lowset = np.partition(slope_med_sm, k_low)[:k_low]
    med_low = np.median(lowset)
    mad_low = np.median(np.abs(lowset - med_low))
    tau = med_low + 3.0 * 1.4826 * mad_low

    # Require stability over a window
    W = max(int(min_window_frac * n), 10)

    # Scan LEFT→RIGHT (high→low RMSD) for FIRST window with slope ≤ τ
    start_idx = None
    for i in range(0, n - W + 1):
        if np.all(slope_med_sm[i:i+W] <= tau):
            start_idx = i
            break
    if start_idx is None:
        start_idx = max(0, n - W)

    # Expand to include full contiguous below-τ region
    j = start_idx + W
    while j < n and slope_med_sm[j] <= tau:
        j += 1
    end_idx = j - 1

    lo_rmsd = distances_desc[start_idx]
    hi_rmsd = distances_desc[end_idx]
    return start_idx, end_idx, lo_rmsd, hi_rmsd, tau, slope_med_sm

# ========= Bootstrap for (point vs plateau) =========

def bootstrap_dg_point_vs_plateau(pmf_desc, bound_idx, plateau_start_idx, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    unbound_vals = pmf_desc[plateau_start_idx:]
    unbound_vals = unbound_vals[~np.isnan(unbound_vals)]
    bound_val = pmf_desc[bound_idx]
    if np.isnan(bound_val) or len(unbound_vals) == 0:
        return np.nan, np.nan
    dg = []
    for _ in range(n_boot):
        u = rng.choice(unbound_vals, size=len(unbound_vals), replace=True)
        dg.append(np.mean(u) - bound_val)
    return float(np.mean(dg)), float(np.std(dg))

# ========= Main =========

def main():
    # Physical params
    k = 40.0           # kcal/mol/Å^2 (match your TMD)
    T = 310.0          # K
    kB = 0.0019872041  # kcal/mol/K
    kT = kB * T

    logfiles = sorted(glob.glob("*.log"))
    if not logfiles:
        print("❌ No log files found.")
        return

    # Pass 1: read all logs, keep first-step RMSD, collect RMSD for grid bounds
    traj_data = []       # list of (rmsds, forces, filename)
    first_rmsds = []
    all_rmsd_pool = []

    for logfile in logfiles:
        steps, rmsds, forces = read_tmd_log(logfile)
        if len(rmsds) == 0:
            continue
        traj_data.append((rmsds, forces, logfile))
        first_rmsds.append(rmsds[0])
        all_rmsd_pool.extend(rmsds.tolist())

    if not traj_data:
        print("❌ No usable work data.")
        return

    # Data-driven RMSD grid (ascending for interpolation; later flip for plotting)
    rmsd_min = float(np.nanmin(all_rmsd_pool))
    rmsd_max = float(np.nanmax(all_rmsd_pool))
    grid_asc = np.linspace(rmsd_min, rmsd_max, 500)   # ascending
    distances = grid_asc[::-1]                        # DESC for plot (high→low RMSD)

    # Pass 2: interpolate work to grid (NaN outside each traj’s RMSD span)
    all_work_interp_asc = []
    smooth_work_asc = []

    for rmsds, forces, logfile in traj_data:
        drmsd = np.gradient(rmsds)
        work = compute_work(forces, k, drmsd)

        order = np.argsort(rmsds)
        xr = rmsds[order]
        yr = work[order]

        try:
            # No extrapolation: values outside each traj span are NaN
            interp_func = interp1d(xr, yr, bounds_error=False, fill_value=np.nan, assume_sorted=True)
            wi = interp_func(grid_asc)
            all_work_interp_asc.append(wi)
            smooth_work_asc.append(savgol_safe(wi, window=11, poly=3))
        except Exception as e:
            print(f"⚠️ Interpolation failed for {logfile}: {e}")

    if not all_work_interp_asc:
        print("❌ No usable work data after interpolation.")
        return

    all_work_interp_asc = np.array(all_work_interp_asc).T  # (n_points × n_traj)
    smooth_work_asc = np.array(smooth_work_asc).T

    # Flip to descending RMSD for plotting convention (high→low)
    all_work_desc = all_work_interp_asc[::-1, :]
    smooth_work_desc = smooth_work_asc[::-1, :]

    # PMFs (NaN-aware Jarzynski)
    pmf_raw_desc        = jarzynski_equality_nanaware(all_work_desc, kT)
    pmf_smooth_desc     = savgol_safe(pmf_raw_desc, window=11, poly=3)
    pmf_w_smooth_desc   = jarzynski_equality_nanaware(smooth_work_desc, kT)

    # Visual band (work-based SE; indicative)
    with np.errstate(invalid='ignore'):
        counts = np.sum(~np.isnan(all_work_desc), axis=1)
        pmf_sem_desc = np.nanstd(all_work_desc, axis=1) / np.sqrt(np.maximum(counts, 1))

    # === Data-driven regions ===
    # Bound = single point from first TMD steps (robust)
    bound_rmsd = float(np.median(first_rmsds))
    bound_idx = int(np.nanargmin(np.abs(distances - bound_rmsd)))

    # --- FIRST force-based plateau (median |dW/dR| across trajectories) ---
    (force_plat_start_idx,
     force_plat_end_idx,
     force_plat_lo,
     force_plat_hi,
     tau_force,
     slope_med_sm) = detect_first_force_plateau(all_work_desc, distances, min_window_frac=0.05)

    # Unbound mask = this FIRST force-based plateau
    unbound_mask_idx = np.arange(force_plat_start_idx, force_plat_end_idx + 1)

    # === ΔG (bound point vs FIRST force-based plateau) ===
    if np.any(~np.isnan(pmf_smooth_desc[unbound_mask_idx])) and not np.isnan(pmf_smooth_desc[bound_idx]):
        dg_point_plateau = np.nanmean(pmf_smooth_desc[unbound_mask_idx]) - pmf_smooth_desc[bound_idx]
    else:
        dg_point_plateau = np.nan

    if np.any(~np.isnan(pmf_w_smooth_desc[unbound_mask_idx])) and not np.isnan(pmf_w_smooth_desc[bound_idx]):
        dg_point_plateau_w = np.nanmean(pmf_w_smooth_desc[unbound_mask_idx]) - pmf_w_smooth_desc[bound_idx]
    else:
        dg_point_plateau_w = np.nan

    # Bootstrap within FIRST force plateau (start_idx used by function)
    dg_boot_mean, dg_boot_std = bootstrap_dg_point_vs_plateau(
        pmf_smooth_desc, bound_idx, force_plat_start_idx, n_boot=1000, seed=42
    )

    # Standard state correction (1 M)
    def standard_state_correction(temp_K):
        R = 1.9872041e-3  # kcal/mol·K
        return -R * temp_K * np.log(1 / 1660.0)  # = +R T ln(1660)

    dg_std_corr = standard_state_correction(T)
    dg_point_plateau_std = dg_point_plateau + dg_std_corr if not np.isnan(dg_point_plateau) else np.nan

    # === Representative trajectory (by plateau mean over this window) ===
    pmf_plateau_mean = np.nanmean(pmf_raw_desc[unbound_mask_idx])
    traj_plateau_means = np.array([
        np.nanmean(all_work_desc[unbound_mask_idx, j]) for j in range(all_work_desc.shape[1])
    ])
    closest_idx = int(np.nanargmin(np.abs(traj_plateau_means - pmf_plateau_mean)))
    closest_file = logfiles[closest_idx] if 0 <= closest_idx < len(logfiles) else "N/A"
    print(f"\n🔍 Closest trajectory to PMF (FIRST force plateau mean): {closest_file}")

    # === Output ===
    print("\n===== Data-driven ΔG (Bound point vs FIRST force-based plateau) =====")
    print(f"Bound (KC) point RMSD (median first steps):   {bound_rmsd:.3f} Å  (idx {bound_idx})")
    print(f"Unbound (FIRST force plateau) RMSD range:     [{force_plat_lo:.3f}, {force_plat_hi:.3f}] Å")
    print(f"FIRST force plateau τ (|dW/dR|):              {tau_force:.5f} kcal/mol/Å")
    print(f"ΔG (Smoothed PMF):                            {dg_point_plateau: .2f} kcal/mol")
    print(f"ΔG (PMF from smoothed work):                  {dg_point_plateau_w: .2f} kcal/mol")
    print(f"Bootstrap ΔG (mean ± sd):                     {dg_boot_mean: .2f} ± {dg_boot_std:.2f} kcal/mol")
    print(f"Standard-state correction (1 M):              {dg_std_corr: .2f} kcal/mol")
    print(f"ΔG corrected to 1 M:                          {dg_point_plateau_std: .2f} kcal/mol")

    # === Plot (start from bound point only) ===
    plt.figure(figsize=(8, 6))

    # Slice everything to start from the bound point
    d_plot            = distances[bound_idx:]
    pmf_raw_plot      = pmf_raw_desc[bound_idx:]
    pmf_smooth_plot   = pmf_smooth_desc[bound_idx:]
    pmf_w_plot        = pmf_w_smooth_desc[bound_idx:]
    pmf_sem_plot      = pmf_sem_desc[bound_idx:]
    smooth_work_plot  = smooth_work_desc[bound_idx:, :]
    closest_traj_plot = all_work_desc[bound_idx:, closest_idx]

    # Smoothed work traces (gray)
    for j in range(smooth_work_plot.shape[1]):
        plt.plot(d_plot, smooth_work_plot[:, j], color='gray', alpha=0.2)

    # PMFs
    plt.plot(d_plot, pmf_raw_plot,      label="Raw PMF",                  linewidth=2)
    plt.plot(d_plot, pmf_smooth_plot,   label="Smoothed PMF",             linewidth=2)
    plt.plot(d_plot, pmf_w_plot,        label="PMF (from smoothed work)", linewidth=2)

    # Visual SE band (work-based, indicative)
    lo = pmf_smooth_plot - pmf_sem_plot
    hi = pmf_smooth_plot + pmf_sem_plot
    plt.fill_between(d_plot, lo, hi, alpha=0.18, label="±SE (work-based)")

    # Bound = vertical line (left edge of plot)
    plt.axvline(bound_rmsd, linestyle='--', linewidth=2, label='Bound point (KC)')

    # Shade the FIRST force-based plateau (not the whole low-RMSD tail)
    plt.axvspan(force_plat_lo, force_plat_hi, alpha=0.14, label='First force-based plateau')

    # Representative trajectory
    plt.plot(d_plot, closest_traj_plot, linewidth=2, label='Closest trajectory')

    plt.gca().invert_xaxis()  # high RMSD left (KC), low RMSD right (monomers)
    plt.xlabel(r"RMSD ($\AA$)", fontsize=15)
    plt.ylabel("Energy (kcal/mol)", fontsize=15)
    plt.title("PMF and ΔG (data-driven bound point & first force plateau)", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pmf_plot.png", dpi=400, bbox_inches="tight")
    plt.show()

    # === Optional: slope diagnostics figure ===
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(distances, slope_med_sm, linewidth=2, label="Median |dW/dR| across trajectories")
    plt.axhline(tau_force, linestyle="--", linewidth=2, label=f"τ = {tau_force:.3g}")
    plt.axvline(force_plat_lo, linestyle="--", linewidth=2, label="Plateau start (first)")
    plt.axvline(force_plat_hi, linestyle="--", linewidth=2, label="Plateau end (first)")
    plt.gca().invert_xaxis()
    plt.xlabel(r"RMSD ($\AA$)")
    plt.ylabel(r"|dW/dR| (kcal mol$^{-1} \AA^{-1}$)")
    plt.title("First force-based plateau detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("force_plateau_diagnostics.png", dpi=400, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
