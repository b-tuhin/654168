# =====================================================================
# CELL 3 - ALL REPORT PLOTS (every run). Run AFTER Cell 2. Changes nothing in Cell 1 or Cell 2.
# Cell 1 and Cell 2 keep drawing the easy-to-read graphs. This cell adds:
#   1. sensor traces with the stops shaded
#   2. top-down path of every run (+ ideal path for a device test)
#   3. distance from the start vs time
#   4. height (Z) vs time
#   Device test only (needs the ideal path, i.e. 'stops found' = 5 for the square):
#   5. ERROR VS DISTANCE at the stops, every run (marked stops hidden: they are not independent)
#   6. ERROR VS TIME: (a) error at each stop vs time, (b) distance to the ideal path vs time (continuous)
#   7. DRIFT ANALYSIS: error ~ c * distance^n for the runs with no outside aid (printed + plotted)
#   8. REPORT TABLE: mean stop error, final stop error, return-to-start, tier result
# EuRoC: Cell 2 already draws the truth-based graphs (trajectory, error vs time, error vs distance, drift).
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

SHOW_SENSORS      = True
SHOW_ALL_PATHS    = True
SHOW_DISTANCE     = True
SHOW_HEIGHT       = True
SHOW_STOP_ERRORS  = True          # plots 5-8 (device test only)
PATH_ZOOM_ON      = "auto"        # run the top-down plot is zoomed on
HIDE_IN_TOPDOWN   = ("raw", "bias removed", "bias + Mahony", "bias + ZUPT")
TIME_PLOT_RUNS    = ("all combined", "ESKF (no aiding)", "leg de-drift (Mahony)", "leg de-drift (ESKF)",
                     "ESKF + corner marks", "leg de-drift + corner marks")   # runs drawn in plot 6
NO_AID_RUNS       = ("raw", "bias removed", "bias + Mahony", "bias + ZUPT", "all combined",
                     "ESKF (no aiding)", "leg de-drift (Mahony)", "leg de-drift (ESKF)")   # used for the drift fit

_runs = dict(runs_all)
_geo = globals().get("geo", None) if results["source"] == "device" else None
_mark_stops = set(globals().get("mark_stops", []) or [])
_mark_runs = set(globals().get("mark_runs", ()) or ())
_t = results["t"]
_cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_color = {n: _cols[i % len(_cols)] for i, n in enumerate(_runs)}
_stops = results.get("stops", [])


def _zoom_run():
    if PATH_ZOOM_ON in _runs:
        return PATH_ZOOM_ON
    for n in ("leg de-drift (ESKF)", "ESKF (no aiding)", "all combined"):
        if n in _runs:
            return n
    return list(_runs)[-1]


def _poly_dist(P, W):
    """Distance from each point of P to the ideal path (uses Cell 2's helper if it exists)."""
    f = globals().get("dist_to_polyline")
    if f is not None:
        return f(P, W)
    best = np.full(len(P), np.inf)
    for a, b in zip(W[:-1], W[1:]):
        ab = b - a; L2 = float(ab @ ab)
        s = np.clip(((P - a) @ ab) / max(L2, 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(P - (a + s[:, None] * ab), axis=1))
    return best


def _stop_errors(name):
    """Errors at stops 1..n. Stops that were given to the filter as a mark are NaN (not an independent test)."""
    pos = _runs[name][0]
    W, arr = _geo["W"], _geo["arr"]
    e = np.array([np.linalg.norm(pos[arr[j]] - W[j]) for j in range(1, len(W))])
    if name in _mark_runs:
        for j in _mark_stops:
            e[j - 1] = np.nan
    return e


# 1. sensors -----------------------------------------------------------
if SHOW_SENSORS:
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for i, c in enumerate("bgr"):
        axs[0].plot(_t, results["acc"][:, i], c, lw=0.8, label="acc " + "xyz"[i])
        axs[1].plot(_t, results["gyro"][:, i], c, lw=0.8, label="gyro " + "xyz"[i])
    for a, b in _stops:
        for ax in axs:
            ax.axvspan(_t[a], _t[b], color="gray", alpha=0.2)
    axs[0].set_ylabel("m/s^2"); axs[1].set_ylabel("rad/s"); axs[1].set_xlabel("time (s)")
    axs[0].set_title("Sensors" + (" (gray = the stops that were found)" if len(_stops) else ""))
    axs[0].legend(ncol=3, fontsize=8); axs[1].legend(ncol=3, fontsize=8)
    plt.tight_layout(); plt.show()

# 2. top-down, every run -----------------------------------------------
if SHOW_ALL_PATHS:
    zr = _zoom_run()
    pz = _runs[zr][0]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    if _geo is not None:
        W = _geo["W"]
        ax.plot(W[:, 0], W[:, 1], "k-o", lw=3, ms=8, label="ideal path (ground truth)")
        for j in range(len(W)):
            ax.annotate(str(j), (W[j, 0], W[j, 1]), textcoords="offset points", xytext=(7, 7))
        pts = np.vstack([W[:, :2], pz[:, :2]])
    else:
        pts = pz[:, :2]
    for n, (pos, _) in _runs.items():
        if n in HIDE_IN_TOPDOWN:
            continue
        ax.plot(pos[:, 0], pos[:, 1], color=_color[n], lw=1.3, label=n)
    ax.plot(0, 0, "ko", ms=6)
    lo, hi = pts.min(0), pts.max(0)
    m = 0.4 * max((hi - lo).max(), 1.0)
    ax.set_xlim(lo[0] - m, hi[0] + m); ax.set_ylim(lo[1] - m, hi[1] + m); ax.set_aspect("equal")
    ax.set_title("Top-down path of every run (zoomed on '%s'; runs that drift off to hundreds of metres are hidden)" % zr, fontsize=9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.show()

# 3. distance from the start vs time ----------------------------------
if SHOW_DISTANCE:
    fig, ax = plt.subplots(figsize=(11, 5))
    for n, (pos, _) in _runs.items():
        ax.plot(_t, np.maximum(np.linalg.norm(pos - pos[0], axis=1), 1e-3), color=_color[n], lw=1.2, label=n)
    for a, b in _stops:
        ax.axvspan(_t[a], _t[b], color="gray", alpha=0.12)
    ax.set_yscale("log"); ax.set_xlabel("time (s)"); ax.set_ylabel("distance from the start (m)")
    ax.set_title("Distance from the start vs time, every run (this is NOT an error plot)"); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()

# 4. height vs time -----------------------------------------------------
if SHOW_HEIGHT:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for n, (pos, _) in _runs.items():
        if n in ("raw", "bias removed"):
            continue
        ax.plot(_t, pos[:, 2] - pos[0, 2], color=_color[n], lw=1.2, label=n)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("height change (m)")
    ax.set_title("Height (Z) vs time (floor test: it should stay near 0; raw and bias-removed runs left out)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()

# 5-8. device test with an ideal path ----------------------------------
if SHOW_STOP_ERRORS and _geo is not None:
    W, arr, cum, total = _geo["W"], _geo["arr"], _geo["cum"], _geo["total"]
    t_stops = np.array([_t[arr[j]] for j in range(1, len(W))])
    errs = {n: _stop_errors(n) for n in _runs}

    # 5. error vs distance (marked stops hidden)
    fig, ax = plt.subplots(figsize=(11, 5))
    for n, e in errs.items():
        ax.plot(cum[1:], e, "-o", color=_color[n], lw=1.2, label=n)      # NaN = a gap, not a fake zero
    ax.plot(cum[1:], 0.03 * cum[1:], color="gray", ls="--", label="3% of distance walked")
    ax.axhline(1.0, color="gray", ls=":", label="1 m")
    ax.axhline(0.3, color="gray", ls="-.", label="0.3 m (bonus)")
    ax.set_yscale("log"); ax.set_xlabel("distance walked along the ideal path (m)"); ax.set_ylabel("error at the stop (m)")
    ax.set_title("ERROR VS DISTANCE at each stop, every run (stops given to a filter as a mark are left out)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()

    # 6. error vs time
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for n in TIME_PLOT_RUNS:
        if n not in errs:
            continue
        axs[0].plot(t_stops, errs[n], "-o", color=_color[n], lw=1.2, label=n)
        d = _poly_dist(_runs[n][0], W)
        axs[1].plot(_t, np.maximum(d, 1e-3), color=_color[n], lw=1.1, label=n)
    axs[0].set_yscale("log"); axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("error at the stop (m)")
    axs[0].set_title("ERROR VS TIME (a): error at each stop"); axs[0].axhline(1.0, color="gray", ls=":")
    axs[1].set_yscale("log"); axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("distance to the ideal path (m)")
    axs[1].set_title("ERROR VS TIME (b): distance to the ideal path (cross-track only,\nit does not see being too far ahead or behind on the path)", fontsize=9)
    for a, b in _stops:
        axs[1].axvspan(_t[a], _t[b], color="gray", alpha=0.12)
    for ax in axs:
        ax.legend(fontsize=7)
    plt.tight_layout(); plt.show()

    # 7. drift analysis: error ~ c * d^n at the stops, no-aid runs only
    print("\nDRIFT ANALYSIS (no outside aid): error at the stops ~ c * (distance walked)^n")
    print("  fitted on only %d stops, so treat n as a rough guide, not a precise number." % (len(W) - 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in NO_AID_RUNS:
        if n not in errs:
            continue
        e, d = errs[n], cum[1:]
        ok = np.isfinite(e) & (e > 1e-6)
        if ok.sum() >= 3:
            nexp, lc = np.polyfit(np.log(d[ok]), np.log(e[ok]), 1)
            print("  %-26s n = %5.2f   (n~1: steady velocity error, n~2: constant acceleration error, n<~0.5: error is bounded)" % (n, nexp))
            ax.loglog(d, e, "o", color=_color[n]); ax.loglog(d, np.exp(lc) * d ** nexp, "-", color=_color[n], label="%s  n=%.2f" % (n, nexp))
    ax.set_xlabel("distance walked (m)"); ax.set_ylabel("error at the stop (m)"); ax.set_title("Drift: error vs distance (log-log), fitted lines")
    ax.legend(fontsize=7); plt.tight_layout(); plt.show()

    # 8. report table
    print("\nREPORT TABLE (stops given to a filter as a mark are left out of the mean)")
    print("%-28s %12s %8s %12s %14s %10s %10s" % ("run", "mean stop m", "mean %", "final stop m", "return-to-start", "Core<1m", "Main<=3%"))
    core_idx = [j for j in range(1, len(W)) if _t[_stops[j][0]] <= _t[_stops[0][1]] + 30.0] or [1]
    for n, e in errs.items():
        pos = _runs[n][0]
        ce = e[[j - 1 for j in core_idx]]
        core = "n/a(mark)" if not np.isfinite(ce).any() else ("PASS" if np.nanmax(ce) < 1.0 else "fail")
        main_ = "PASS" if e[-1] <= 0.03 * total else "fail"
        print("%-28s %12.2f %7.1f%% %12.2f %14.2f %10s %10s" % (n, np.nanmean(e), 100 * np.nanmean(e) / total, e[-1],
                                                                np.linalg.norm(pos[-1] - pos[0]), core, main_))
    print("Test path is %.2f m, so 3%% = %.2f m (the problem statement uses a 50 m path, 1.5 m)." % (total, 0.03 * total))
    print("De-drift runs fix each leg when you stop at the next corner, so they are not real-time. Say so in the report.")
elif SHOW_STOP_ERRORS and results["source"] == "device":
    print("No error plots: the ideal-path scoring was skipped (stop count was wrong, see Cell 1/Cell 2 messages).")

if results["source"] == "euroc":
    print("EuRoC: the truth-based graphs (trajectory vs truth, error vs time, error vs distance, bias, tilt, drift) come from Cell 2.")
