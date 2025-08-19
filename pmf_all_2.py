"""
Post-processing script for Targeted Molecular Dynamics (TMD) trajectories
to estimate a Potential of Mean Force (PMF) along RMSD and compute binding
free energy (ΔG_bind) with data-driven region detection.

Key updates vs your version:
- Derives the RMSD grid from the data (no hard-coded 12.5→1.5).
- Bound region = a single point from the first TMD steps (robust median),
  shown as a vertical line on the plot.
- Unbound region = detected automatically as a plateau on the low-RMSD end
  using a robust slope threshold (median + 3*MAD).
- ΔG is computed as (mean PMF over detected plateau) − (PMF at bound point).
- Bootstrap resamples WITHIN the plateau only (bound is a single point).
- Representative trajectory chosen by plateau mean, not single endpoint.
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
    # Your heuristic: cumulative "-1/2 k (ΔR)^2 dR" in energy units
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

# ========= Plateau detection (data-driven) =========

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

def detect_plateau(distances_desc, pmf_desc, min_window_frac=0.05):
    """
    Detect a plateau at the low-RMSD end (right side of descending arrays).
    Returns (plateau_start_idx, plateau_lo_rmsd, plateau_hi_rmsd, tau)
    """
    # Guard
    if len(distances_desc) < 10:
        return 0, distances_desc[0], distances_desc[-1], np.nan

    # Smooth for stability
    pmf_sm = savgol_safe(pmf_desc, window=max(11, len(pmf_desc)//20*2+1), poly=3)

    # Absolute slope
    grad = np.abs(np.gradient(pmf_sm, distances_desc))

    n = len(distances_desc)
    W = max(int(min_window_frac * n), 10)

    # Use last 20% (lowest RMSD tail) for robust slope baseline
    tail = grad[max(0, int(0.8 * n)):] if n >= 10 else grad
    tau = np.median(tail) + 3.0 * 1.4826 * mad(tail)  # median + 3*MAD

    # Scan from right (low RMSD) to left for first W-length window with slope ≤ tau
    plateau_start_idx = None
    for i in range(n - W, -1, -1):
        window_ok = np.all(grad[i:i+W] <= tau)
        # also require enough non-NaNs in the pmf window
        if window_ok and np.sum(~np.isnan(pmf_desc[i:i+W])) >= max(5, W//2):
            plateau_start_idx = i
            break
    if plateau_start_idx is None:
        plateau_start_idx = max(0, n - W)

    plateau_lo = distances_desc[plateau_start_idx]
    plateau_hi = distances_desc[-1]  # right end, lowest RMSD
    return plateau_start_idx, plateau_lo, plateau_hi, tau

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

def detect_plateau_2(distances_desc, pmf_desc, min_window_frac=0.05):
    """
    Detect a plateau at the low-RMSD end (right side of descending arrays).
    Returns:
      plateau_start_idx, plateau_lo_rmsd, plateau_hi_rmsd, tau, pmf_sm, grad
    """
    if len(distances_desc) < 10:
        pmf_sm = pmf_desc.copy()
        grad = np.abs(np.gradient(pmf_sm, distances_desc)) if len(distances_desc) > 1 else np.array([np.nan])
        return 0, distances_desc[0], distances_desc[-1], np.nan, pmf_sm, grad

    # Smooth PMF for stable slope estimates
    pmf_sm = savgol_safe(pmf_desc, window=max(11, len(pmf_desc)//20*2+1), poly=3)

    # Absolute slope
    grad = np.abs(np.gradient(pmf_sm, distances_desc))

    n = len(distances_desc)
    W = max(int(min_window_frac * n), 10)

    # Robust threshold from the last 20% (lowest RMSD tail)
    tail = grad[max(0, int(0.8 * n)):] if n >= 10 else grad
    def _mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med))
    tau = np.median(tail) + 3.0 * 1.4826 * _mad(tail)

    # Scan from right (low RMSD) to left for first W-length window with slope ≤ tau
    plateau_start_idx = None
    for i in range(n - W, -1, -1):
        window_ok = np.all(grad[i:i+W] <= tau)
        has_data = np.sum(~np.isnan(pmf_desc[i:i+W])) >= max(5, W//2)
        if window_ok and has_data:
            plateau_start_idx = i
            break
    if plateau_start_idx is None:
        plateau_start_idx = max(0, n - W)

    plateau_lo = distances_desc[plateau_start_idx]
    plateau_hi = distances_desc[-1]
    return plateau_start_idx, plateau_lo, plateau_hi, tau, pmf_sm, grad



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
            # Fill NaN outside the observed RMSD range (avoid extrapolation)
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

    # Flip to descending RMSD for your plotting convention (high→low)
    all_work_desc = all_work_interp_asc[::-1, :]
    smooth_work_desc = smooth_work_asc[::-1, :]

    # PMFs (NaN-aware Jarzynski)
    pmf_raw_desc        = jarzynski_equality_nanaware(all_work_desc, kT)
    pmf_smooth_desc     = savgol_safe(pmf_raw_desc, window=11, poly=3)
    pmf_w_smooth_desc   = jarzynski_equality_nanaware(smooth_work_desc, kT)

    # Simple visual band (work-based SE, for visualization only)
    with np.errstate(invalid='ignore'):
        pmf_sem_desc = np.nanstd(all_work_desc, axis=1) / np.sqrt(np.sum(~np.isnan(all_work_desc), axis=1))

    # === Data-driven regions ===
    # Bound = single point from first TMD steps (robust)
    bound_rmsd = float(np.median(first_rmsds))
    bound_idx = int(np.nanargmin(np.abs(distances - bound_rmsd)))

    # Unbound = detected plateau (from smoothed PMF)
    plateau_start_idx, plateau_lo, plateau_hi, tau = detect_plateau(distances, pmf_smooth_desc, min_window_frac=0.05)
    unbound_mask_idx = np.arange(plateau_start_idx, len(distances))

    # === ΔG (bound point vs plateau) ===
    # Using smoothed PMF
    if np.any(~np.isnan(pmf_smooth_desc[unbound_mask_idx])) and not np.isnan(pmf_smooth_desc[bound_idx]):
        dg_point_plateau = np.nanmean(pmf_smooth_desc[unbound_mask_idx]) - pmf_smooth_desc[bound_idx]
    else:
        dg_point_plateau = np.nan

    # Using PMF-from-smoothed-work (optional diagnostic)
    if np.any(~np.isnan(pmf_w_smooth_desc[unbound_mask_idx])) and not np.isnan(pmf_w_smooth_desc[bound_idx]):
        dg_point_plateau_w = np.nanmean(pmf_w_smooth_desc[unbound_mask_idx]) - pmf_w_smooth_desc[bound_idx]
    else:
        dg_point_plateau_w = np.nan

    # Bootstrap within plateau
    dg_boot_mean, dg_boot_std = bootstrap_dg_point_vs_plateau(
        pmf_smooth_desc, bound_idx, plateau_start_idx, n_boot=1000, seed=42
    )

    # Standard state correction (1 M)
    def standard_state_correction(temp_K):
        R = 1.9872041e-3  # kcal/mol·K
        return -R * temp_K * np.log(1 / 1660.0)  # = +R T ln(1660)

    dg_std_corr = standard_state_correction(T)
    dg_point_plateau_std = dg_point_plateau + dg_std_corr if not np.isnan(dg_point_plateau) else np.nan

    # === Representative trajectory (by plateau mean) ===
    # Compare each trajectory’s plateau-mean work to PMF plateau-mean
    pmf_plateau_mean = np.nanmean(pmf_raw_desc[unbound_mask_idx])
    traj_plateau_means = np.array([
        np.nanmean(all_work_desc[unbound_mask_idx, j]) for j in range(all_work_desc.shape[1])
    ])
    closest_idx = int(np.nanargmin(np.abs(traj_plateau_means - pmf_plateau_mean)))
    closest_file = logfiles[closest_idx] if 0 <= closest_idx < len(logfiles) else "N/A"
    print(f"\n🔍 Closest trajectory to PMF (plateau mean): {closest_file}")

    # === Output ===
    print("\n===== Data-driven ΔG (Bound = first-step point, Unbound = detected plateau) =====")
    print(f"Bound (KC) point RMSD (median of first steps): {bound_rmsd:.3f} Å  (grid idx {bound_idx})")
    print(f"Unbound plateau RMSD range:                   [{plateau_lo:.3f}, {plateau_hi:.3f}] Å")
    print(f"Plateau slope threshold τ (|dF/dR|):          {tau:.5f} kcal/mol/Å")
    print(f"ΔG (Smoothed PMF):                            {dg_point_plateau: .2f} kcal/mol")
    print(f"ΔG (PMF from smoothed work):                  {dg_point_plateau_w: .2f} kcal/mol")
    print(f"Bootstrap ΔG (mean ± sd):                     {dg_boot_mean: .2f} ± {dg_boot_std:.2f} kcal/mol")
    print(f"Standard-state correction (1 M):              {dg_std_corr: .2f} kcal/mol")
    print(f"ΔG corrected to 1 M:                          {dg_point_plateau_std: .2f} kcal/mol")

    # === Plot ===
    plt.figure(figsize=(8, 6))

    # Slice everything to start from the bound point
    d_plot   = distances[bound_idx:]
    pmf_raw_plot     = pmf_raw_desc[bound_idx:]
    pmf_smooth_plot  = pmf_smooth_desc[bound_idx:]
    pmf_w_plot       = pmf_w_smooth_desc[bound_idx:]
    pmf_sem_plot     = pmf_sem_desc[bound_idx:]
    smooth_work_plot = smooth_work_desc[bound_idx:, :]
    closest_traj_plot = all_work_desc[bound_idx:, closest_idx]

    # Smoothed work traces (gray)
    for j in range(smooth_work_plot.shape[1]):
        plt.plot(d_plot, smooth_work_plot[:, j], color='gray', alpha=0.2)

    # PMFs
    plt.plot(d_plot, pmf_raw_plot,        label="Raw PMF",              linewidth=2)
    plt.plot(d_plot, pmf_smooth_plot,     label="Smoothed PMF",         linewidth=2)
    plt.plot(d_plot, pmf_w_plot,          label="PMF (from smoothed work)", linewidth=2)

    # Visual SE band (work-based, indicative)
    lo = pmf_smooth_plot - pmf_sem_plot
    hi = pmf_smooth_plot + pmf_sem_plot
    plt.fill_between(d_plot, lo, hi, alpha=0.18, label="±SE (work-based)")

    # Bound = vertical line (now at left edge of plot)
    plt.axvline(bound_rmsd, linestyle='--', linewidth=2, label='Bound point (KC)')

    # Unbound plateau shading (still relative to sliced arrays)
    plt.axvspan(distances[plateau_start_idx], distances[-1], alpha=0.12,
                label='Unbound (detected plateau)')

    # Representative trajectory
    plt.plot(d_plot, closest_traj_plot, linewidth=2, label='Closest trajectory')

    plt.gca().invert_xaxis()
    plt.xlabel(r"RMSD ($\AA$)", fontsize=15)
    plt.ylabel("Energy (kcal/mol)", fontsize=15)
    plt.title("PMF and ΔG (data-driven bound point & plateau)", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pmf_plot.png", dpi=400, bbox_inches="tight")
    plt.show()

    # Unbound = detected plateau (from smoothed PMF) + diagnostics
    (plateau_start_idx,
     plateau_lo,
     plateau_hi,
     tau,
     pmf_sm_for_diag,
     grad_abs) = detect_plateau_2(distances, pmf_smooth_desc, min_window_frac=0.05)

    unbound_mask_idx = np.arange(plateau_start_idx, len(distances))

    # === Print a small table around the plateau start ===
    pad = 8
    i0 = max(0, plateau_start_idx - pad)
    i1 = min(len(distances), plateau_start_idx + pad + 1)
    print("\n--- Plateau diagnostics (around start) ---")
    print(" idx |   RMSD(Å)  |   PMF_smooth  | |dF/dR|  | below τ | in plateau ")
    for i in range(i0, i1):
        below = grad_abs[i] <= tau
        in_plat = (i >= plateau_start_idx)
        print(f"{i:4d} | {distances[i]:10.3f} | {pmf_smooth_desc[i]:11.3f} | {grad_abs[i]:7.4f} |"
              f" {str(below):>7s} | {str(in_plat):>10s}")

    # === Save full diagnostics to CSV ===
    # Columns: RMSD, PMF_smooth, abs_slope, below_tau(0/1), in_plateau(0/1)
    below_tau = (grad_abs <= tau).astype(int)
    in_plateau = (np.arange(len(distances)) >= plateau_start_idx).astype(int)
    diag_mat = np.column_stack([distances, pmf_smooth_desc, grad_abs, below_tau, in_plateau])
    np.savetxt("plateau_diagnostics.csv",
               diag_mat,
               fmt="%.6f,%.6f,%.6f,%d,%d",
               header="RMSD_A,PMF_smooth,abs_slope_kcal_per_mol_per_A,below_tau,in_plateau",
               comments="")
    print("\n📄 Wrote plateau diagnostics to plateau_diagnostics.csv")

    # === Quick diagnostic plot of |dF/dR| with τ and plateau start ===
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(distances, grad_abs, linewidth=2, label="|dF/dR| (from smoothed PMF)")
    plt.axhline(tau, linestyle="--", linewidth=2, label=f"τ threshold = {tau:.4g}")
    plt.axvline(distances[plateau_start_idx], linestyle="--", linewidth=2, label="Plateau start")
    plt.gca().invert_xaxis()  # keep your high→low RMSD convention
    plt.xlabel(r"RMSD ($\AA$)")
    plt.ylabel(r"|dF/dR| (kcal mol$^{-1} \AA^{-1}$)")
    plt.title("Plateau detection diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plateau_diagnostics.png", dpi=400, bbox_inches="tight")
    plt.show()





if __name__ == "__main__":
    main()
