# =====================================================================
# CELL 2 - SCORING + aided runs.
#   EuRoC : simple Kalman filter and ESKF with SIMULATED sensors (ground truth + noise), scored against ground truth.
#   device: the TEST you chose in Cell 1 ("square" / "slide"): ideal path from the tape / ruler, taped floor marks
#           as a declared aid, scored at every stop.
# Run AFTER Cell 1.  Everything lands in the same tables and plots.
# =====================================================================

# ---- EuRoC settings ----
MOTION_START = 5.2            # EuRoC: second at which the drone really starts to move
ALIGN_SECONDS = 30.0          # EuRoC: heading of the un-aided runs is matched using the first 30 s of motion
USE_AIDING   = True           # EuRoC: True = also run the aided filters (simple KF + ESKF variants)
ANCHOR_EVERY = 5.0            # s between simulated position fixes (UWB / fiducial tag style)
ANCHOR_SIGMA = 0.10           # m, noise of each fix
USE_BARO     = True           # simulated barometer (height only) - used by the simple KF only
BARO_RATE_HZ = 5.0
BARO_SIGMA   = 0.10           # m
SIGMA_ACC    = 0.2            # simple KF: how much to distrust the accelerometer
SIGMA_BIAS   = 0.02           # simple KF: how fast the lumped acceleration bias may change
SEED         = 0
RUN_ESKF_VARIANTS = True      # False = skip the ESKF aided runs (faster), keep only the simple KF
ESKF_MAIN    = "ESKF + position fixes"   # run used for the 1-sigma and bias plots
ESKF_TRUTH_START = False      # DEV ONLY. True = the aided ESKF runs start with the TRUE tilt and accel bias (from the
                              # ground truth). State it in the report if used.
TUNE = False                  # True = also try several noise inflation factors (uses truth: development only)

# ---- device settings (the TEST itself is chosen in Cell 1) ----
USE_MARKS    = True           # FIX 3: use the taped floor marks as a declared aid (extra runs)
MARK_SIGMA   = 0.05           # m, how exactly you place the device on a mark (tape tolerance)
MARK_STOPS   = "intermediate" # "intermediate" = marks at the corners only; the LAST stop (back at the start) is then an
                              # honest, independent test.  "all" = marks at every stop (stop errors are then ~0).
MARK_AFTER_S = 0.5            # s after a stop begins, the mark is applied
ARRIVE_S     = 0.3            # s after a stop begins, the position is read for scoring
KNOWN_PATH_LENGTH_M = None    # TEST = "none": length of your taped path in metres, if you know it
SHOW_SIMPLE_PLOT = True      # the easy plot: ideal path + raw + final estimate(s)
SHOW_FULL_PLOTS  = True      # the detailed plots: every run vs the ideal path, height, error vs time / distance
PLOT_RUNS = ("raw", "leg de-drift (ESKF)", "ESKF + corner marks")   # the runs drawn in the device plot (the ideal path is always drawn)


# ---------------------------------------------------------------------
# EuRoC: simple Kalman filter, state = position (3), velocity (3), acceleration bias (3), room frame
# ---------------------------------------------------------------------
def _kf_update(x, P, H, z, Rm):
    S = H @ P @ H.T + Rm
    K = np.linalg.solve(S, H @ P).T            # = P H^T S^-1  (S is symmetric)
    x = x + K @ (z - H @ x)
    P = (np.eye(len(x)) - K @ H) @ P
    return x, 0.5 * (P + P.T)


def kalman_aided(t, acc, Rs, g, p0, anchors=None, baro=None, sigma_acc=0.2, sigma_bias=0.02):
    """anchors = (times, positions N x 3, sigma);  baro = (times, heights, sigma).
    Orientation Rs comes from the Mahony run; the filter learns the leftover acceleration bias itself."""
    N = len(t)
    g_vec = np.array([0.0, 0.0, g])
    I3 = np.eye(3)
    x = np.zeros(9); x[0:3] = p0
    P = np.diag([0.1 ** 2] * 3 + [0.05 ** 2] * 3 + [1.0 ** 2] * 3)
    pos = np.zeros((N, 3)); pos[0] = p0
    ia = ib = 0

    for k in range(1, N):
        dt = t[k] - t[k - 1]
        a = Rs[k] @ acc[k] - g_vec                        # room-frame acceleration, gravity removed

        F = np.eye(9)
        F[0:3, 3:6] = I3 * dt
        F[0:3, 6:9] = -0.5 * dt * dt * I3
        F[3:6, 6:9] = -dt * I3
        x = F @ x + np.concatenate([0.5 * dt * dt * a, dt * a, np.zeros(3)])
        Gm = np.zeros((9, 3)); Gm[0:3] = 0.5 * dt * dt * I3; Gm[3:6] = dt * I3
        Q = (Gm @ Gm.T) * (sigma_acc ** 2 / dt)
        Q[6:9, 6:9] += sigma_bias ** 2 * dt * I3
        P = F @ P @ F.T + Q

        if anchors is not None and ia < len(anchors[0]) and t[k] >= anchors[0][ia]:
            H = np.zeros((3, 9)); H[:, 0:3] = I3
            x, P = _kf_update(x, P, H, anchors[1][ia], anchors[2] ** 2 * I3)
            ia += 1
        if baro is not None and ib < len(baro[0]) and t[k] >= baro[0][ib]:
            H = np.zeros((1, 9)); H[0, 2] = 1.0
            x, P = _kf_update(x, P, H, np.array([baro[1][ib]]), np.array([[baro[2] ** 2]]))
            ib += 1

        pos[k] = x[0:3]
    return pos


def R_from_accel_with_truth_yaw(acc_still, R_truth):
    """Tilt from the still accelerometer; heading from the room frame (anchors live in the room frame)."""
    R_tilt = R_from_accel(acc_still)
    R_rel = R_truth @ R_tilt.T
    yaw = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ R_tilt


# ---------------------------------------------------------------------
# EuRoC: simulated aiding sensors, truth + noise. One random generator, so every filter sees the SAME data.
# ---------------------------------------------------------------------
def make_sim_sensors(t, t_gt, pos_gt):
    rng = np.random.default_rng(SEED)
    t_lo, t_hi = max(t[0], t_gt[0]), min(t[-1], t_gt[-1])
    tr = lambda tq: np.column_stack([np.interp(tq, t_gt, pos_gt[:, i]) for i in range(3)])
    a_t = np.arange(t_lo + ANCHOR_EVERY, t_hi, ANCHOR_EVERY)
    a_p = tr(a_t) + rng.normal(0.0, ANCHOR_SIGMA, (len(a_t), 3))
    r_t = np.arange(t_lo, t_hi, 1.0 / UWB_RATE)
    d = np.linalg.norm(tr(r_t)[:, None, :] - UWB_ANCHORS[None, :, :], axis=2)
    r = d + rng.normal(0.0, UWB_SIGMA, d.shape)
    r = r + (rng.random(d.shape) < UWB_OUTLIER_P) * rng.uniform(0.3, 1.5, d.shape)
    b_t = np.arange(t_lo, t_hi, 1.0 / BARO_RATE_HZ)
    b_z = np.interp(b_t, t_gt, pos_gt[:, 2]) + rng.normal(0.0, BARO_SIGMA, len(b_t))
    return (a_t, a_p, ANCHOR_SIGMA), (r_t, r, UWB_SIGMA), (b_t, b_z, BARO_SIGMA)


def load_truth_bias(folder):
    """Truth accel and gyro bias (EuRoC ground-truth file). For checking only."""
    first = int(_read(os.path.join(folder, "mav0", "imu0", "data.csv"))[0][0])
    rows = _read(os.path.join(folder, "mav0", "state_groundtruth_estimate0", "data.csv"))
    t_b = np.array([(int(r[0]) - first) * 1e-9 for r in rows])
    arr = np.array([[float(x) for x in r[11:17]] for r in rows])
    return t_b, arr[:, 0:3], arr[:, 3:6]       # t, gyro bias, accel bias


def _room_frame_start(results, t_gt, pos_gt, q_gt):
    """Start in the room frame: position and heading from the room, tilt from the accelerometer."""
    m = (t_gt >= results["t_start"]) & (t_gt <= results["still_end"])
    p0 = pos_gt[m].mean(axis=0) if m.any() else pos_gt[0]
    q = q_gt[np.argmin(np.abs(t_gt - 0.5 * (results["t_start"] + results["still_end"])))]
    R_truth = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return p0, R_from_accel_with_truth_yaw(results["acc_still"], R_truth), R_truth


def run_aided(results, folder):
    """EuRoC. Returns (runs, infos). runs: name -> (pos, Rs).  infos: name -> ESKF diagnostics."""
    t, gyro, acc = results["t"], results["gyro"], results["acc"]
    gyro_bias, g_est = results["gyro_bias"], results["g_est"]
    t_gt, pos_gt, q_gt = load_truth(folder)
    p0, R0, R_truth = _room_frame_start(results, t_gt, pos_gt, q_gt)
    eskf_extra = {}
    if ESKF_TRUTH_START:
        print("ESKF ORACLE START ON (dev only): tilt and accel bias taken from ground truth")
        R0 = R_truth
        eskf_extra = dict(ba0=results["acc_still"] - R_truth.T @ np.array([0.0, 0.0, g_est]))
    fixes, ranges, baro = make_sim_sensors(t, t_gt, pos_gt)

    new, infos = {}, {}

    # --- the simple Kalman filter (orientation from Mahony, 9 states) ---
    print("running: aided (simple KF) ...")
    _, Rs = navigate(t, gyro, acc, R0, p0, gyro_bias=gyro_bias, g=g_est, kp=KP, ki=KI)
    pos = kalman_aided(t, acc, Rs, g_est, p0, anchors=fixes, baro=baro if USE_BARO else None,
                       sigma_acc=SIGMA_ACC, sigma_bias=SIGMA_BIAS)
    new["aided (simple KF)"] = (pos, Rs)

    # --- the ESKF variants (15 states, learns both biases and tilt, outlier gate) ---
    if RUN_ESKF_VARIANTS:
        fs = 1.0 / np.median(np.diff(t))
        gyro_lp, acc_lp = lowpass(gyro, fs, LOWPASS_HZ), lowpass(acc, fs, LOWPASS_HZ)
        jobs = {
            "ESKF + position fixes":          dict(g_=gyro, a_=acc, fixes=fixes),
            "ESKF + fixes, low-pass IMU":     dict(g_=gyro_lp, a_=acc_lp, fixes=fixes, acc_for_still=acc),
            "ESKF + fixes, no gravity upd.":  dict(g_=gyro, a_=acc, fixes=fixes, use_grav=False),
            "ESKF + UWB ranges (gated)":      dict(g_=gyro, a_=acc, ranges=ranges),
            "ESKF + UWB ranges (NO gate)":    dict(g_=gyro, a_=acc, ranges=ranges, use_gate=False),
        }
        for name, kw in jobs.items():
            print("running:", name, "...")
            kw = dict(kw)
            g_, a_ = kw.pop("g_"), kw.pop("a_")
            pos, Rs_, info = eskf(t, g_, a_, R0, p0, g_est, gyro_bias, **eskf_extra, **kw)
            new[name], infos[name] = (pos, Rs_), info

    print("aided runs: %d simulated position fixes (every %.1f s, sigma %.2f m), UWB %d anchors at %.0f Hz "
          "(sigma %.2f m, %.0f%% blocked)%s" %
          (len(fixes[0]), ANCHOR_EVERY, ANCHOR_SIGMA, len(UWB_ANCHORS), UWB_RATE, UWB_SIGMA,
           100 * UWB_OUTLIER_P,
           " | simple KF also gets a simulated barometer %.0f Hz, sigma %.2f m" % (BARO_RATE_HZ, BARO_SIGMA)
           if USE_BARO else ""))
    return new, infos


# ---------------------------------------------------------------------
# EuRoC scoring
# ---------------------------------------------------------------------
def _yaw_matrix(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def score_euroc(results, runs, folder, motion_start, align_seconds, already_aligned=(), infos=None):
    infos = infos or {}
    t = results["t"]
    t_gt, pos_gt, _ = load_truth(folder)

    ok = (t >= t_gt[0]) & (t <= t_gt[-1])              # only where truth exists
    tt = t[ok]
    truth = np.column_stack([np.interp(tt, t_gt, pos_gt[:, i]) for i in range(3)])
    dist = path_length(truth)
    n = np.searchsorted(tt, motion_start + align_seconds)
    i10 = np.argmin(np.abs(tt - (motion_start + 10)))
    i30 = np.argmin(np.abs(tt - (motion_start + 30)))
    after = tt >= motion_start

    print("Truth path length: %.2f m | truth start-to-end: %.2f m" % (dist[-1], np.linalg.norm(truth[-1] - truth[0])))
    print("%-32s %9s %9s %10s %12s %14s" % ("run", "err@10s", "err@30s", "end err", "end err %", "return-to-start"))

    aligned, errs = {}, {}
    for name, (pos, _) in runs.items():
        est = pos[ok]
        if name in already_aligned:                    # already in the room frame, no matching needed
            a = est
        else:
            best = None                                # try every heading, keep the best match
            for deg in range(360):
                cand = (est[:n] - est[0]) @ _yaw_matrix(deg).T + truth[0]
                e = np.linalg.norm(cand - truth[:n], axis=1).mean()
                if best is None or e < best[0]:
                    best = (e, deg)
            a = (est - est[0]) @ _yaw_matrix(best[1]).T + truth[0]
        err = np.linalg.norm(a - truth, axis=1)
        aligned[name], errs[name] = a, err
        rts = np.linalg.norm(est[-1] - est[0])
        print("%-32s %9.2f %9.2f %10.2f %11.1f%% %14.2f" %
              (name, err[i10], err[i30], err[-1], 100 * err[-1] / dist[-1], rts))
    print("(err@10s / err@30s = metres of error that many seconds after motion starts)")
    print("(un-aided runs: heading matched to truth over the first %.0f s of motion, a best-case alignment)" % align_seconds)

    # --- error statistics after the motion starts ---
    print("\nERROR STATISTICS after motion start (metres)")
    print("%-32s %8s %8s %8s %9s %9s" % ("run", "mean", "p95", "max", "rejected", "still-upd"))
    for name, err in errs.items():
        e = err[after]
        inf = infos.get(name)
        rj = "%9d" % inf["rejected"] if inf else "%9s" % "-"
        st = "%9d" % inf["still"] if inf else "%9s" % "-"
        print("%-32s %8.2f %8.2f %8.2f %s %s" % (name, e.mean(), np.percentile(e, 95), e.max(), rj, st))
    print("p95 = 95% of the time the error is below this.  rejected = measurements thrown out by the outlier gate")
    print("(gravity + fixes + ranges).  still-upd = zero-velocity updates applied.  '-' = not reported by that run.")

    # --- tier check against the problem statement ---
    print("\nTIER CHECK")
    print("%-32s %-24s %-26s %-22s" % ("run", "Core: err@30s < 1 m", "Main: end err <= 3% path", "Bonus: max err < 0.3 m"))
    for name, err in errs.items():
        core = "PASS" if err[i30] < 1.0 else "fail"
        main_ = "PASS" if err[-1] <= 0.03 * dist[-1] else "fail"
        mx = err[after].max()
        bonus = ("PASS" if mx < 0.3 else "fail") + " (max %.2f)" % mx
        print("%-32s %-24s %-26s %-22s" % (name, core + " (%.2f m)" % err[i30], main_ + " (%.2f m)" % err[-1], bonus))
    print("Aided runs (simple KF and ESKF) use SIMULATED anchors / UWB / barometer (truth + noise). State this in the report.")

    # --- plots: one pair for the un-aided runs, one pair for the aided runs ---
    groups = [("UN-AIDED", [k for k in runs if k not in already_aligned]),
              ("AIDED (simulated sensors)", [k for k in runs if k in already_aligned])]
    for gname, names in groups:
        if not names:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
        axs[0].plot(truth[:, 0], truth[:, 1], "k", lw=3, label="ground truth")
        for name in names:
            axs[0].plot(aligned[name][:, 0], aligned[name][:, 1], label=name, lw=1.2)
        lo, hi = truth[:, :2].min(0), truth[:, :2].max(0)
        m = 0.5 * max((hi - lo).max(), 1.0)
        axs[0].set_xlim(lo[0] - m, hi[0] + m); axs[0].set_ylim(lo[1] - m, hi[1] + m); axs[0].set_aspect("equal")
        axs[0].set_title("%s: top-down trajectory vs ground truth" % gname)
        axs[0].set_xlabel("X (m)"); axs[0].set_ylabel("Y (m)"); axs[0].legend(fontsize=8)
        axs[1].plot(tt, truth[:, 2], "k", lw=3, label="ground truth")
        for name in names:
            axs[1].plot(tt, aligned[name][:, 2], label=name, lw=1.2)
        axs[1].set_ylim(truth[:, 2].min() - 1.5, truth[:, 2].max() + 1.5)
        axs[1].set_title("%s: height Z vs time" % gname); axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("Z (m)")
        axs[1].legend(fontsize=8)
        plt.tight_layout()

        fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
        for name in names:
            axs[0].plot(tt, np.maximum(errs[name], 1e-3), label=name, lw=1.0)
            axs[1].plot(dist, np.maximum(errs[name], 1e-3), label=name, lw=1.0)
        axs[0].axvline(motion_start, color="gray", ls=":"); axs[0].set_title("%s: position error vs time" % gname)
        axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("error (m)")
        axs[1].plot(dist, np.maximum(0.03 * dist, 1e-3), color="gray", ls="--", label="3% of distance so far")
        axs[1].axhline(1.0, color="gray", ls=":", label="1 m")
        axs[1].set_title("%s: position error vs distance travelled" % gname)
        axs[1].set_xlabel("distance travelled (m)")
        for ax in axs:
            ax.set_yscale("log"); ax.legend(fontsize=8)
        plt.tight_layout()
    plt.show()

    # --- drift analysis: error ~ c * t^n  (un-aided runs only) ---
    print("\nDRIFT ANALYSIS: error grows roughly like (time since motion start)^n")
    sel = tt > motion_start + 5
    tm = tt[sel] - motion_start
    for name, err in errs.items():
        if name in already_aligned:
            continue
        e = np.maximum(err[sel], 1e-3)
        nexp = np.polyfit(np.log(tm), np.log(e), 1)[0]
        print("  %-32s n = %.2f" % (name, nexp))
    print("  n near 1: steady velocity error.  n near 2: constant acceleration error (bias, or gravity")
    print("  leaking in through a tilt error).  n above 2: the tilt error itself keeps growing.")
    return errs, tt, ok


# ---------------------------------------------------------------------
# EuRoC: extra ESKF plots and checks
# ---------------------------------------------------------------------
def plot_eskf_extras(results, runs, infos, errs, tt, ok, folder, folder_main=None, ba_ls=None):
    main = folder_main or ESKF_MAIN
    if main not in infos:
        return
    t = results["t"]
    mk = tt >= MOTION_START

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    for n in ("ESKF + position fixes", "ESKF + UWB ranges (gated)", "ESKF + UWB ranges (NO gate)"):
        if n in errs:
            axs[0].plot(tt[mk], np.maximum(errs[n][mk], 1e-3), label=n)
    axs[0].plot(tt[mk], infos[main]["pstd"][ok][mk], "k--", label="filter's own 1-sigma (%s)" % main)
    axs[0].set_yscale("log"); axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("error (m)")
    axs[0].set_title("ESKF error vs time; dashed = what the filter believes its error is"); axs[0].legend(fontsize=8)
    if "ESKF (no aiding)" in errs and main in errs:
        axs[1].plot(tt[mk], np.maximum(errs["ESKF (no aiding)"][mk], 1e-3), label="ESKF, no aiding")
        axs[1].plot(tt[mk], np.maximum(errs[main][mk], 1e-3), label=main)
    axs[1].set_yscale("log"); axs[1].set_xlabel("time (s)"); axs[1].set_title("What aiding adds"); axs[1].legend()
    plt.tight_layout()

    try:
        tb, bw_true, ba_true = load_truth_bias(folder)
        fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
        ba_est, bg_est = infos[main]["ba"], infos[main]["bg"]
        for i, c in enumerate("bgr"):
            axs[0].plot(t, ba_est[:, i], c, label="estimated " + "xyz"[i])
            axs[0].plot(tb, ba_true[:, i], c + "--", alpha=0.6, label="truth file " + "xyz"[i])
            if ba_ls is not None:
                axs[0].axhline(ba_ls[i], color=c, ls=":", lw=1.6, label="data-consistent " + "xyz"[i])
            axs[1].plot(t, bg_est[:, i], c, label="estimated " + "xyz"[i])
            axs[1].plot(tb, bw_true[:, i], c + "--", alpha=0.6, label="truth " + "xyz"[i])
        axs[0].set_title("Accelerometer bias (m/s^2)"); axs[1].set_title("Gyro bias (rad/s)")
        for ax in axs:
            ax.set_xlabel("time (s)"); ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
    except Exception as e:
        print("(bias plot skipped: %s)" % e)
    plt.show()


def truth_consistent_bias(results, folder, g=G):
    """Which accel bias does the RECORDING itself support?  Uses truth orientation + truth acceleration (second
    derivative of the truth position) to predict what the accelerometer should read, then compares with what it
    did read:  acc = s * R^T (a_true + g) + b  (least squares per axis).  Uses ground truth: checking only."""
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Slerp
    t, acc = results["t"], results["acc"]
    t_gt, pos_gt, q_gt = load_truth(folder)
    dtg = float(np.median(np.diff(t_gt)))
    win = int(round(0.4 / dtg)) // 2 * 2 + 1                              # ~0.4 s, odd
    a_gt = savgol_filter(pos_gt, win, 3, deriv=2, delta=dtg, axis=0)
    ok = (t >= max(t_gt[0], MOTION_START)) & (t <= t_gt[-1])
    Rr = Slerp(t_gt, Rot.from_quat(q_gt[:, [1, 2, 3, 0]]))(t[ok]).as_matrix()
    a_t = np.column_stack([np.interp(t[ok], t_gt, a_gt[:, i]) for i in range(3)])
    f_pred = np.einsum("nji,nj->ni", Rr, a_t + np.array([0.0, 0.0, g]))     # R^T (a + g)
    resid = acc[ok] - f_pred
    b_only = resid.mean(axis=0)
    fit = [np.linalg.lstsq(np.column_stack([f_pred[:, i], np.ones(len(f_pred))]), acc[ok][:, i], rcond=None)[0]
           for i in range(3)]
    return b_only, np.array(fit), resid, t[ok]


def print_bias_check(results, folder, infos, name=None):
    name = name or ESKF_MAIN
    b_ls, fit, _, _ = truth_consistent_bias(results, folder)
    tb, _, ba_true = load_truth_bias(folder)
    ba_file = ba_true.mean(axis=0)
    ba_est = infos[name]["ba"][-1] if name in infos else np.full(3, np.nan)
    print("\nACCEL BIAS CHECK (m/s^2)              x        y        z")
    print("  truth file, mean            %8.3f %8.3f %8.3f" % tuple(ba_file))
    print("  truth-consistent (LS)       %8.3f %8.3f %8.3f   <- what this recording's accelerometer actually shows" % tuple(b_ls))
    print("  ESKF value (end)            %8.3f %8.3f %8.3f" % tuple(ba_est))
    print("  per-axis scale error        %7.2f%% %7.2f%% %7.2f%%   (acc = s*f + b fit; bias in that fit: %s)" %
          (*(100 * (fit[:, 0] - 1.0)), np.round(fit[:, 1], 3)))
    d_est, d_file = abs(ba_est[1] - b_ls[1]), abs(ba_file[1] - b_ls[1])
    if d_est < d_file:
        print("  -> y: the ESKF is CLOSER to what the data shows than the truth-file value, so the truth-file bias does")
        print("     not describe this recording well (it is itself a fitted estimate). The 'offset' is not a filter error.")
    else:
        print("  -> y: the data supports the truth-file value, so the ESKF y bias is genuinely under-estimated")
        print("     (tilt/bias ambiguity at start-up).")


def plot_tilt_bias_diagnostics(results, runs, infos, folder, name=None):
    """Orientation error vs truth (left) and accel bias error (right)."""
    from scipy.spatial.transform import Slerp
    name = name or ESKF_MAIN
    if name not in runs:
        return
    t = results["t"]
    t_gt, _, q_gt = load_truth(folder)
    ok = (t >= t_gt[0]) & (t <= t_gt[-1])
    R_true = Slerp(t_gt, Rot.from_quat(q_gt[:, [1, 2, 3, 0]]))(t[ok]).as_matrix()      # SciPy wants x, y, z, w
    Rs = runs[name][1][ok]
    e = np.degrees(Rot.from_matrix(Rs @ np.transpose(R_true, (0, 2, 1))).as_rotvec())  # world-frame error rotation
    tilt = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", Rs[:, :, 2], R_true[:, :, 2]), -1.0, 1.0)))
    tb, _, ba_true = load_truth_bias(folder)
    ba_t = np.column_stack([np.interp(t[ok], tb, ba_true[:, i]) for i in range(3)])
    ba_err = infos[name]["ba"][ok] - ba_t
    mv = t[ok] >= MOTION_START

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
    for i, (c, lab) in enumerate(zip("bgr", ("about x (roll-like)", "about y (pitch-like)", "about z (heading)"))):
        axs[0].plot(t[ok], e[:, i], c, label=lab)
    axs[0].plot(t[ok], tilt, "k--", label="total tilt error")
    axs[0].axhline(0, color="gray", lw=0.5)
    axs[0].set_title("Orientation error vs truth (%s)" % name); axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("degrees")
    axs[0].legend(fontsize=8)
    for i, c in enumerate("bgr"):
        axs[1].plot(t[ok], ba_err[:, i], c, label="bias error " + "xyz"[i])
    axs[1].axhline(0, color="gray", lw=0.5)
    axs[1].set_title("Accelerometer bias error (estimate - truth file)"); axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("m/s^2")
    axs[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()
    print("%s: mean tilt error %.2f deg | final accel-bias error x %.3f  y %.3f  z %.3f m/s^2" %
          (name, tilt[mv].mean(), *ba_err[-1]))


def tune_noise(results, folder):
    """Optional: how much to inflate the datasheet noise (uses truth, development only)."""
    t_gt, pos_gt, q_gt = load_truth(folder)
    t = results["t"]
    p0, R0, _ = _room_frame_start(results, t_gt, pos_gt, q_gt)
    fixes, _, _ = make_sim_sensors(t, t_gt, pos_gt)
    ok = (t >= max(t_gt[0], MOTION_START)) & (t <= t_gt[-1])
    truth = np.column_stack([np.interp(t[ok], t_gt, pos_gt[:, i]) for i in range(3)])
    print("\n%10s %10s %8s %8s" % ("acc_x", "gyro_x", "mean", "max"))
    for ax_ in (5, 20, 50, 100):
        for gx in (3, 10, 30):
            pos, _, _ = eskf(t, results["gyro"], results["acc"], R0, p0, results["g_est"], results["gyro_bias"],
                             fixes=fixes, acc_x=ax_, gyr_x=gx)
            e = np.linalg.norm(pos[ok] - truth, axis=1)
            print("%10d %10d %8.2f %8.2f" % (ax_, gx, e.mean(), e.max()))


# ---------------------------------------------------------------------
# DEVICE: the test's ideal path in the room frame, the corner marks, and the scoring at every stop
# ---------------------------------------------------------------------
def _rot2(psi_deg):
    c, s = np.cos(np.radians(psi_deg)), np.sin(np.radians(psi_deg))
    return np.array([[c, -s], [s, c]])


def fit_test_heading(p_stops, wps_by_turn):
    """Which way did the first leg go in the room frame?  Compare the direction of every leg of the drift-corrected
    path with the direction of the same leg of the ideal path; average the angle differences (weighted by leg length).
    quality: 1 = all legs agree, 0 = no agreement. Returns (turn, psi_deg, quality)."""
    d = np.diff(np.asarray(p_stops)[:, :2], axis=0)
    best = None
    for turn, w in wps_by_turn.items():
        u = np.diff(w, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(u[:, 1], u[:, 0])
        wt = np.linalg.norm(d, axis=1)
        c = np.sum(wt * np.exp(1j * ang))
        q = float(abs(c) / max(wt.sum(), 1e-9))
        if best is None or q > best[2]:
            best = (turn, float(np.degrees(np.angle(c))), q)
    return best


def device_geometry(results):
    """The ideal path of the chosen TEST placed in the room frame. Returns a dict, or None if it cannot be matched."""
    test, stops, t = results["test"], results["stops"], results["t"]
    if test == "none":
        return None
    turns = ["cw", "ccw"] if (test == "square" and TEST_TURN == "auto") else [TEST_TURN if test == "square" else "cw"]
    wps = {c: test_waypoints(test, c) for c in turns}
    n_wp = len(next(iter(wps.values())))
    if len(stops) != n_wp:
        print("TEST '%s' needs %d stops (start + %d arrivals) but %d were found. The ideal-path scoring and the corner"
              % (test, n_wp, n_wp - 1, len(stops)))
        print("marks are skipped. Tune STOP_MIN_S / STOP_BRIDGE_S in Cell 1, or check that you stopped at every corner.")
        return None
    ref = results["runs"].get("leg de-drift (Mahony)")
    if ref is None:
        print("no leg de-drift run available, so the heading of the test cannot be fitted; scoring skipped")
        return None
    dt = float(np.median(np.diff(t)))
    off = int(round(ARRIVE_S / dt))
    arr = [min(a + off, b) for a, b in stops]
    turn, psi, q = fit_test_heading(ref[0][arr], wps)
    src = "fitted from the data"
    if TEST_HEADING_DEG is not None:
        psi, src = float(TEST_HEADING_DEG), "set by TEST_HEADING_DEG"
    w = wps[turn]
    W = np.column_stack([(_rot2(psi) @ w.T).T, np.zeros(len(w))]) + results["p0"]
    legs = np.linalg.norm(np.diff(W, axis=0), axis=1)
    print("\nIDEAL PATH of the '%s' test in the room frame" % test)
    print("  turn direction: %s | direction of the first leg: %.0f deg (%s) | fit quality %.2f (1 = perfect)" %
          (turn if test == "square" else "-", psi, src, q))
    if q < 0.6 and TEST_HEADING_DEG is None:
        print("  WARNING: the fit quality is low, so the fitted direction may be wrong. If you know how you held the device,")
        print("  set TEST_HEADING_DEG in Cell 1 (device flat, top edge pointing along leg 1: 90) and run again.")
    print("  real path length %.2f m | corner positions (m): %s" % (legs.sum(), "; ".join("(%.2f, %.2f)" % (x, y) for x, y, _ in W)))
    return dict(turn=turn, psi=psi, quality=q, W=W, arr=arr, legs=legs, total=float(legs.sum()),
                cum=np.concatenate([[0.0], np.cumsum(legs)]))


def run_marks(results, geo):
    """FIX 3: the taped floor marks as a declared aid. Two runs:
       'ESKF + corner marks'          : each mark is a position fix (sigma MARK_SIGMA, flat floor => z = start height)
       'leg de-drift + corner marks'  : per-leg correction that also has to arrive exactly at the mark"""
    t, stops, W = results["t"], results["stops"], geo["W"]
    which = list(range(1, len(stops) - 1)) if MARK_STOPS == "intermediate" else list(range(1, len(stops)))
    if not which:
        print("this test has no intermediate stop to put a mark on; marks skipped")
        return {}, {}, []
    dt = float(np.median(np.diff(t)))
    off = int(round(MARK_AFTER_S / dt))
    times = np.array([t[min(stops[j][0] + off, stops[j][1])] for j in which])
    marks = {j: W[j] for j in which}
    new, infos = {}, {}
    print("\nrunning: ESKF + corner marks ...")
    pos, Rs, info = eskf(t, results["gyro"], results["acc"], results["R0"], results["p0"], results["g_est"],
                         results["gyro_bias"], fixes=(times, np.array([W[j] for j in which]), MARK_SIGMA), use_gate=False)
    new["ESKF + corner marks"], infos["ESKF + corner marks"] = (pos, Rs), info
    print("running: leg de-drift + corner marks ...")
    Rs_m = results["runs"]["bias + Mahony"][1]
    a_m = np.einsum("kij,kj->ki", Rs_m, results["acc"]) - np.array([0.0, 0.0, results["g_est"]])
    new["leg de-drift + corner marks"] = (dedrift(t, a_m, stops, results["p0"], marks), Rs_m)
    print("AID USED: taped floor marks at the corners (position fix, sigma %.2f m) + flat floor (z = start height)."
          % MARK_SIGMA)
    print("marks at stops %s of 0..%d; %s" % (which, len(stops) - 1,
          "the last stop (back at the start) is NOT given to the filter, so its error is an honest test"
          if MARK_STOPS == "intermediate" else "every stop is a mark, so stop errors are about zero by design"))
    return new, infos, which


def dist_to_polyline(P, W):
    """3D distance from each point of P (n x 3) to the polyline through the corners W (m x 3)."""
    best = np.full(len(P), np.inf)
    for a, b in zip(W[:-1], W[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            d = np.linalg.norm(P - a, axis=1)
        else:
            s = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
            d = np.linalg.norm(P - (a + s[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def plot_device_summary(results, runs, geo, plot_runs, mark_stops, mark_runs):
    """ONE clean figure: the ideal path (ground truth), the raw run, and the final estimate(s)."""
    t, W, arr, total = results["t"], geo["W"], geo["arr"], geo["total"]
    names = [n for n in plot_runs if n in runs]
    colors = {"raw": "tab:red"}
    palette = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    ax.plot(W[:, 0], W[:, 1], "k-o", lw=3, ms=8, label="ideal path (ground truth)")
    for j in range(len(W)):
        ax.annotate(str(j), (W[j, 0], W[j, 1]), textcoords="offset points", xytext=(7, 7), fontsize=10)
    k = 0
    for n in names:
        if n not in colors:
            colors[n] = palette[k % len(palette)]; k += 1
        pos = runs[n][0]
        ax.plot(pos[:, 0], pos[:, 1], color=colors[n], lw=1.5, label="%s (last stop %.2f m off)" % (
            n, np.linalg.norm(pos[arr[-1]] - W[-1])))
    lo, hi = W[:, :2].min(0), W[:, :2].max(0)
    m = 1.0 * max((hi - lo).max(), 0.5)
    ax.set_xlim(lo[0] - m, hi[0] + m); ax.set_ylim(lo[1] - m, hi[1] + m); ax.set_aspect("equal")
    ax.set_title("Top-down: ideal '%s' path vs estimates" % results["test"])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(fontsize=8, loc="best")
    ax = axs[1]
    for n in names:
        pos = runs[n][0]
        e = np.array([np.linalg.norm(pos[arr[j]] - W[j]) for j in range(1, len(W))])
        ax.plot(geo["cum"][1:], np.maximum(e, 1e-3), "-o", color=colors[n], label=n)
    ax.plot(geo["cum"][1:], 0.03 * geo["cum"][1:], color="gray", ls="--", label="3% of distance walked")
    ax.set_yscale("log"); ax.set_xlabel("distance walked along the ideal path (m)"); ax.set_ylabel("error at the stop (m)")
    ax.set_title("Position error at each stop"); ax.legend(fontsize=8)
    plt.tight_layout()


def plot_device_full(results, runs, geo):
    """The detailed plots: every run against the ideal path, plus error versus time and versus distance."""
    t, arr, W, total = results["t"], geo["arr"], geo["W"], geo["total"]
    n_stop = len(W)
    stop_err = {name: np.array([np.linalg.norm(pos[arr[j]] - W[j]) for j in range(1, n_stop)])
                for name, (pos, _) in runs.items()}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax = axs[0]
    ax.plot(W[:, 0], W[:, 1], "k-o", lw=3, ms=8, label="ideal path (tape / ruler)")
    for j in range(n_stop):
        ax.annotate(str(j), (W[j, 0], W[j, 1]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    for name, (pos, _) in runs.items():
        ax.plot(pos[:, 0], pos[:, 1], lw=1.2, label=name)
    lo, hi = W[:, :2].min(0), W[:, :2].max(0)
    m = 1.0 * max((hi - lo).max(), 0.5)
    ax.set_xlim(lo[0] - m, hi[0] + m); ax.set_ylim(lo[1] - m, hi[1] + m); ax.set_aspect("equal")
    ax.set_title("Top-down: estimates vs the ideal '%s' path (zoomed on the ideal path)" % results["test"])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(fontsize=7)
    ax = axs[1]
    for name, (pos, _) in runs.items():
        ax.plot(t, pos[:, 2], lw=1.0, label=name)
    ax.axhline(W[0, 2], color="k", lw=2, label="flat floor")
    ax.set_ylim(W[0, 2] - 2.0, W[0, 2] + 2.0)
    ax.set_title("Height Z vs time (should stay flat)"); ax.set_xlabel("time (s)"); ax.set_ylabel("Z (m)"); ax.legend(fontsize=7)
    plt.tight_layout()

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    for name, (pos, _) in runs.items():
        dev = dist_to_polyline(pos, W)
        axs[0].plot(t, np.maximum(dev, 1e-3), lw=1.0, label=name)
        axs[1].plot(geo["cum"][1:], np.maximum(stop_err[name], 1e-3), "-o", lw=1.0, label=name)
    for ax in axs:
        ax.axhline(0.03 * total, color="gray", ls="--", label="3% of the real path")
        ax.axhline(1.0, color="gray", ls=":", label="1 m")
        ax.set_yscale("log"); ax.legend(fontsize=7)
    axs[0].set_title("Distance from the ideal path vs time (path deviation)")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("m")
    axs[1].set_title("Position error at each stop vs distance travelled")
    axs[1].set_xlabel("distance travelled along the ideal path (m)"); axs[1].set_ylabel("m")
    plt.tight_layout()


def score_device_test(results, runs, geo, mark_stops, mark_runs):
    t, stops, arr, W, total = results["t"], results["stops"], geo["arr"], geo["W"], geo["total"]
    n_stop = len(W)
    motion_start = t[stops[0][1]]
    print("\n" + "=" * 100)
    print("SCORING against the ideal path of the '%s' test (real path %.2f m)" % (results["test"], total))
    print("=" * 100)
    print("error (m) = distance between the estimated position and the ideal corner, read %.1f s after arriving at each stop" % ARRIVE_S)
    head = "%-28s" % "run" + "".join("%9s" % ("stop %d" % j) for j in range(1, n_stop)) + "%10s %12s %12s %12s" % ("final %", "farthest", "path est.", "return@end")
    print(head)
    stop_err = {}
    for name, (pos, _) in runs.items():
        e = np.array([np.linalg.norm(pos[arr[j]] - W[j]) for j in range(1, n_stop)])
        stop_err[name] = e
        cells = ""
        for j in range(1, n_stop):
            star = "*" if (name in mark_runs and j in mark_stops) else " "
            cells += "%8.2f%s" % (e[j - 1], star)
        far = np.linalg.norm(pos - pos[0], axis=1).max()
        print("%-28s%s%9.1f%% %12.2f %12.2f %12.2f" % (name, cells, 100 * e[-1] / total, far,
                                                       path_length(pos)[-1], np.linalg.norm(pos[-1] - pos[0])))
    print("* = this stop was given to the filter as a mark (its error is not an independent test).")
    print("final % = error at the LAST stop (back at the start) as a percentage of the real path length.")
    print("farthest = farthest the estimate got from the start (ideal: %.2f m); path est. = estimated path length (ideal %.2f m)."
          % (np.linalg.norm(W - W[0], axis=1).max(), total))

    # error as a percentage of the path, and the length of each leg
    print("\nERROR AS A PERCENTAGE OF THE REAL PATH (%.2f m)" % total)
    print("%-28s %14s %12s %12s" % ("run", "mean stop err", "mean %", "final %"))
    for name, e in stop_err.items():
        print("%-28s %12.2f m %11.1f%% %11.1f%%" % (name, e.mean(), 100 * e.mean() / total, 100 * e[-1] / total))
    print("mean % = average of the stop errors, as a percentage of the path (a fairer number than the final stop alone).")
    print("\nLEG LENGTHS (m) between stops. Ideal: %s" % "  ".join("%.2f" % x for x in geo["legs"]))
    for name in ("ESKF (no aiding)", "leg de-drift (Mahony)", "leg de-drift (ESKF)", "ESKF + corner marks"):
        if name in runs:
            p = runs[name][0][arr]
            print("  %-26s %s" % (name, "  ".join("%.2f" % x for x in np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))))

    # tier check
    print("\nTIER CHECK (same targets as the problem statement, applied to this test)")
    print("%-28s %-26s %-28s %-24s" % ("run", "Core: stops in first 30 s < 1 m", "Main: final stop <= 3% of path", "Bonus: all stops < 0.3 m"))
    core_idx = [j for j in range(1, n_stop) if t[stops[j][0]] <= motion_start + 30.0] or [1]
    for name, e in stop_err.items():
        core_err = max(e[j - 1] for j in core_idx)
        core = ("PASS" if core_err < 1.0 else "fail") + " (%.2f m)" % core_err
        main_ = ("PASS" if e[-1] <= 0.03 * total else "fail") + " (%.2f m vs %.2f m)" % (e[-1], 0.03 * total)
        bonus = ("PASS" if e.max() < 0.3 else "fail") + " (max %.2f m)" % e.max()
        print("%-28s %-26s %-28s %-24s" % (name, core, main_, bonus))
    print("Runs with 'corner marks' use a declared aid (taped floor marks). The others use no outside information.")

    if SHOW_SIMPLE_PLOT:
        plot_device_summary(results, runs, geo, PLOT_RUNS, mark_stops, mark_runs)
    if SHOW_FULL_PLOTS:
        plot_device_full(results, runs, geo)
    plt.show()

    # drift summary
    print("\nDRIFT SUMMARY: error at the stops grows with distance walked (no outside aid):")
    for name in ("raw", "bias removed", "bias + Mahony", "all combined", "ESKF (no aiding)",
                 "leg de-drift (Mahony)", "leg de-drift (ESKF)"):
        if name in stop_err:
            print("  %-26s" % name + "  ".join("%.2f m @ %.1f m" % (e, d) for e, d in zip(stop_err[name], geo["cum"][1:])))
    print("  The error grows from stop to stop. Mahony keeps the tilt from drifting; the per-leg correction removes the")
    print("  leftover constant acceleration error in each leg; the marks stop the error from carrying over to the next leg.")


def score_device_generic(results, runs, known_path_m):
    """TEST = 'none': only consistency checks (no ideal path)."""
    print("%-26s %12s %14s %16s %14s" % ("run", "path (m)", "farthest (m)", "return-to-start", "return/path"))
    for name, (pos, _) in runs.items():
        plen = path_length(pos)[-1]
        far = np.linalg.norm(pos - pos[0], axis=1).max()
        rts = np.linalg.norm(pos[-1] - pos[0])
        print("%-26s %12.2f %14.2f %16.2f %13.1f%%" % (name, plen, far, rts, 100 * rts / max(plen, 1e-9)))
    if known_path_m:
        print("Known path length (tape measure): %.2f m. Compare it with the path column." % known_path_m)
    plt.figure(figsize=(6, 6))
    for name, (pos, _) in runs.items():
        plt.plot(pos[:, 0], pos[:, 1], label=name)
        plt.plot(pos[0, 0], pos[0, 1], "ko")
    ref = runs["leg de-drift (Mahony)"][0] if "leg de-drift (Mahony)" in runs else \
        (runs["all combined"][0] if "all combined" in runs else list(runs.values())[0][0])
    lo, hi = ref[:, :2].min(0), ref[:, :2].max(0)
    m = 0.5 * max((hi - lo).max(), 1.0)
    plt.xlim(lo[0] - m, hi[0] + m); plt.ylim(lo[1] - m, hi[1] + m)
    plt.axis("equal"); plt.legend(fontsize=7); plt.title("Device path, zoomed (black dot = start)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.show()


# ---------------------------------------------------------------------
if results["source"] == "euroc":
    runs_all = dict(results["runs"])                   # includes "ESKF (no aiding)" from Cell 1
    aligned_names, infos_all = (), {}
    if results.get("eskf_info") is not None:
        infos_all["ESKF (no aiding)"] = results["eskf_info"]
    if USE_AIDING:
        new_runs, new_infos = run_aided(results, EUROC_FOLDER)
        runs_all.update(new_runs)
        infos_all.update(new_infos)
        aligned_names = tuple(new_runs)
    cell2_errs, cell2_tt, cell2_ok = score_euroc(results, runs_all, EUROC_FOLDER, MOTION_START, ALIGN_SECONDS,
                                                 aligned_names, infos_all)
    if USE_AIDING and RUN_ESKF_VARIANTS:
        ba_ls = truth_consistent_bias(results, EUROC_FOLDER)[0]
        print_bias_check(results, EUROC_FOLDER, infos_all)
        plot_eskf_extras(results, runs_all, infos_all, cell2_errs, cell2_tt, cell2_ok, EUROC_FOLDER, ba_ls=ba_ls)
        plot_tilt_bias_diagnostics(results, runs_all, infos_all, EUROC_FOLDER)
    if TUNE:
        tune_noise(results, EUROC_FOLDER)
else:
    runs_all = dict(results["runs"])
    geo = device_geometry(results)
    if geo is None:
        score_device_generic(results, runs_all, KNOWN_PATH_LENGTH_M)
    else:
        mark_runs, mark_stops = (), []
        if USE_MARKS:
            new_runs, _, mark_stops = run_marks(results, geo)
            runs_all.update(new_runs)
            mark_runs = tuple(new_runs)
        score_device_test(results, runs_all, geo, set(mark_stops), mark_runs)
