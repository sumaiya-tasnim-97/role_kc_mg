"""
Post-processing script for Targeted Molecular Dynamics (TMD) trajectories
to estimate a Potential of Mean Force (PMF) along RMSD and compute several
binding free energy (ΔG_bind) estimates.

Pipeline:
1) Parse TMD log files for step, RMSD, and restraint deviation.
2) Compute trajectory-wise work profiles under a harmonic bias.
3) Interpolate all profiles onto a common RMSD grid and smooth them.
4) Apply Jarzynski’s equality across trajectories to obtain PMF vs RMSD.
5) Compute ΔG_bind using several definitions and a standard-state correction.
6) Identify the trajectory most similar to the PMF endpoint.
7) Plot work traces, PMF curves, and uncertainty bands.

Assumptions:
- Log lines start with 'TMD' and contain step, target RMSD, and current RMSD
  at fixed positions in the split line.
- RMSD decreases from ~12.5 Å (unbound) to ~1.5 Å (bound).
- Harmonic restraint spring constant (k) and temperature (T) match the TMD setup.
"""

"""
    Parse a TMD log file to extract simulation steps, current RMSD, and
    restraint deviation (target RMSD - current RMSD).

    Parameters
    ----------
    logfile : str
        Path to a single TMD log file.

    Returns
    -------
    steps : np.ndarray
        Simulation step numbers.
    rmsds : np.ndarray
        Current RMSD values (reaction coordinate).
    forces : np.ndarray
        Deviation between target and current RMSD, used as 'force' term.

    Notes
    -----
    - Only lines starting with 'TMD' are processed.
    - If a line is malformed, it is skipped with a warning.
"""
    
"""
    Integrate the work profile along RMSD for a trajectory under a harmonic bias.

    Work = -0.5 * k * Σ [ (force)^2 * ΔRMSD ]

    Parameters
    ----------
    forces : np.ndarray
        Deviation between target and current RMSD at each frame.
    k : float
        Harmonic spring constant (energy/Å²).
    drmsd : np.ndarray
        Finite difference of RMSD values along trajectory.

    Returns
    -------
    work : np.ndarray
        Cumulative work profile in kcal/mol.
"""
"""
    Apply Jarzynski’s equality to an ensemble of work profiles to estimate
    free energy change as a function of RMSD.

    F(R) = -kT * ln ⟨ exp( -[ W_i(R) - Wmin(R) ] / kT ) ⟩ + Wmin(R)

    where Wmin is subtracted for numerical stability.

    Parameters
    ----------
    work : np.ndarray, shape (n_points, n_traj)
        Work profiles interpolated on a common RMSD grid.
    kT : float
        Thermal energy (k_B * T) in kcal/mol.

    Returns
    -------
    pmf : np.ndarray
        Potential of mean force along RMSD.
"""
"""
    Apply Savitzky–Golay smoothing to a curve.

    Parameters
    ----------
    y : np.ndarray
        Data to smooth.
    window : int
        Window size (must be odd).
    poly : int
        Polynomial order for smoothing.

    Returns
    -------
    y_smooth : np.ndarray
        Smoothed curve.
"""
"""
    Bootstrap ΔG between bound and unbound regions by resampling PMF points.

    Parameters
    ----------
    pmf : np.ndarray
        Potential of mean force values along RMSD.
    distances : np.ndarray
        RMSD grid points.
    bound_mask : np.ndarray[bool]
        Boolean mask for bound region.
    unbound_mask : np.ndarray[bool]
        Boolean mask for unbound region.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        RNG seed.

    Returns
    -------
    dg_mean : float
        Mean ΔG_bind from bootstrap samples.
    dg_std : float
        Standard deviation of ΔG_bind from bootstrap samples.
"""
"""
    Standard state correction for 1 M binding free energy from a PMF well depth.

    Parameters
    ----------
    temp_K : float
        Temperature in Kelvin.

    Returns
    -------
    correction : float
        ΔG° correction in kcal/mol.

    Notes
    -----
    Volume of 1 M in Å³ is ~1660 Å³.
"""
"""
    Identify the trajectory whose final work value is closest to the PMF endpoint.

    Parameters
    ----------
    pmf : np.ndarray
        PMF along RMSD.
    work : np.ndarray
        Work profiles for each trajectory.
    filenames : list[str]
        List of trajectory log filenames.

    Returns
    -------
    closest_filename : str
        Filename of the closest-matching trajectory.
"""



import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

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
    V0 = 1 / (6.022e23 * 1e-24)  # ~1660 Å³ for 1 M
    return -R * temp_K * np.log(1 / 1660)

def find_closest_file(pmf, work, filenames):
    """
    Find the filename corresponding to the trajectory whose final work value
    is closest to the final value of the PMF.
    """
    final_pmf_value = pmf[-1]
    work_diff = np.abs(work[-1, :] - final_pmf_value)
    closest_index = np.argmin(work_diff)
    return filenames[closest_index]


def main():
    k = 40.0
    T = 310
    kB = 0.0019872041
    kT = kB * T
    rmsd_uniform = np.linspace(12.5, 1.5, 500)

    logfiles = sorted(glob.glob("*.log"))
    if not logfiles:
        print("❌ No log files found.")
        return

    all_work_interp = []
    smooth_work = []

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
            smooth_work.append(smooth(work_interp, window=11, poly=3))
        except Exception as e:
            print(f"⚠️ Interpolation failed for {logfile}: {e}")

    if not all_work_interp:
        print("❌ No usable work data.")
        return

    all_work_interp = np.array(all_work_interp).T
    smooth_work = np.array(smooth_work).T
    pmf_raw = jarzynski_equality(all_work_interp, kT)
    pmf_smooth = smooth(pmf_raw, window=11, poly=3)
    pmf_w_smooth = jarzynski_equality(smooth_work,kT)
    pmf_std = np.std(all_work_interp, axis=1) / np.sqrt(all_work_interp.shape[1])

    # === ΔG_bind region masks ===
    distances = rmsd_uniform
    bound_mask = distances == np.min(distances)
    unbound_mask = (distances > 10.0) & (distances < 12.0)

    grad = np.abs(np.gradient(pmf_smooth, distances))
    plateau_start_index = np.argmax(grad < 0.1)  # 0.5 kcal/mol/Å is an example threshold
    dg_auto_plateau = pmf_smooth[plateau_start_index] - np.min(pmf_smooth)

    grad = np.abs(np.gradient(pmf_w_smooth, distances))
    plateau_start_index = np.argmax(grad < 0.1)  # 0.5 kcal/mol/Å is an example threshold
    dg_auto_plateau_2 = pmf_w_smooth[plateau_start_index] - np.min(pmf_w_smooth)

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
    
    print("\n===== Estimated Binding Free Energy from PMF =====")
    print(f"1. Pointwise ΔG_bind (plateau - min):       {dg_pointwise_2:.2f} kcal/mol")
    print(f"2. Region-averaged ΔG_bind:                 {dg_avg_2:.2f} kcal/mol")
    print(f"3. Max - Min PMF range:                     {dg_maxmin_2:.2f} kcal/mol")
    print(f"4. Bootstrapped ΔG_bind:                    {dg_boot_mean_2:.2f} ± {dg_boot_std:.2f} kcal/mol")
    print(f"5. Standard-state correction (1 M):         {dg_std_corr_2:.2f} kcal/mol")
    print(f"6. ΔG_bind corrected to standard state:     {(dg_avg_2 + dg_std_corr_2):.2f} kcal/mol")
    print(f"7. Detect plateau automatically:            {dg_auto_plateau_2:.2f} kcal/mol")

    # === Plot ===
    plt.figure(figsize=(8, 6))
#    for i in range(all_work_interp.shape[1]):
#        plt.plot(distances, all_work_interp[:, i], color='gray', alpha=0.2)
    for i in range(smooth_work.shape[1]):
        plt.plot(distances, smooth_work[:, i], color='gray', alpha=0.2)
    plt.plot(distances, pmf_raw, label="Raw PMF", color='orange', linewidth=2)
    plt.plot(distances, pmf_smooth, label="Smoothed PMF", color='blue', linewidth=2)
    plt.plot(distances, pmf_w_smooth, label="Smoothed W PMF", color='yellow', linewidth=2)
    plt.fill_between(distances, pmf_smooth - pmf_std, pmf_smooth + pmf_std,
                     color='blue', alpha=0.2, label="Std. deviation")

    # Highlight binding/unbinding regions
    plt.axvspan(1.0, 5.0, color='green', alpha=0.1, label='Unbound region (Monomers)')
    plt.axvspan(12.49, 12.5, color='red', alpha=0.1, label='Bound region (KC)')
    # Highlight closest trajectory
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

