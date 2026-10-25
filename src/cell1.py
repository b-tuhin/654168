# =====================================================================
# CELL 1 - IMU dead reckoning + ESKF (no outside aiding).
# Ground truth is NOT used unless USE_TRUTH_FOR_START = True (dev only).
# The ESKF helpers/settings live here so Cell 2 can reuse them for the aided runs.
# =====================================================================
import csv, os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import butter, lfilter, lfilter_zi

G = 9.81

# ---------------------------------------------------------------------
# SETTINGS - edit these
# ---------------------------------------------------------------------
SOURCE = "device"         # "euroc" (the drone dataset)  or  "device" (a phone / board / any IMU you recorded)
TEST = "square"           # device only: "square" | "slide" | "none".  Picks the recording AND the ideal path.
USE_CALIBRATION = True    # False = raw sensor data.  True = apply the calibration table first (device only).
USE_TRUTH_FOR_START = False   # DEV ONLY, EuRoC only. True = start tilt + accel bias come from ground truth.
RUN_ESKF = True           # also run the error-state Kalman filter (no outside aiding)

# --- EuRoC data ---
try:
    EUROC_FOLDER = data_root                  # made by your setup cell
except NameError:
    EUROC_FOLDER = "/content/euroc_small"     # folder that contains mav0

# --- device data (used when SOURCE = "device") ---
# One recording per test: the .zip the app saved, or the folder that holds its csv files.
DEVICE_RECORDINGS = {
    "square": "/content/drive/MyDrive/colab_upload/phone/walk.zip",     # the 2.34 m square walk
    "slide":  "/content/drive/MyDrive/colab_upload/phone/slide.zip",    # the slide test (record it, then edit this path)
    "none":   "/content/drive/MyDrive/colab_upload/phone/walk.zip",     # any other recording
}
DEVICE_T_START = {"square": 8.0, "slide": 2.0, "none": 2.0}   # s: a moment inside the first still period of each recording
                                                             # (if you change it, check that "stops found" below still says 5 for the square)
DEVICE_ACC_NAME   = "TotalAcceleration.csv"       # accelerometer WITH gravity (still, it reads +9.8 upward)
DEVICE_GYRO_NAME  = "GyroscopeUncalibrated.csv"   # raw gyro (we remove its bias ourselves)
DEVICE_TIME_COL   = "seconds_elapsed"             # name of the time column in the CSV header (exactly as written)
DEVICE_TIME_UNIT  = "s"                           # "ns", "us", "ms" or "s"
DEVICE_ACC_COLS   = ("x", "y", "z")               # read by NAME (the file lists them as z,y,x)
DEVICE_GYRO_COLS  = ("x", "y", "z")
DEVICE_ACC_FACTOR  = 1.0
DEVICE_GYRO_FACTOR = 1.0
DEVICE_RATE_HZ     = 60                           # accel and gyro are resampled onto one clock at this rate
DEVICE_ACC_ND, DEVICE_GYR_ND = 2.24e-3, 5.1e-4    # measured from the six-position test (calibration cell prints these)

# --- the two tests: the ideal path is known from the tape / ruler ---
SQUARE_SIDE_M  = 2.34     # square test: side of the taped square (m)
SLIDE_LENGTH_M = 1.00     # slide test: EDIT to the real length you slid (m)
SLIDE_PASSES   = 2        # slide test: 1 = one way, 2 = out and back, 4 = out-back-out-back
TEST_TURN      = "auto"   # square test: "cw" (right turns, seen from above), "ccw", or "auto" (decided from the data)
TEST_HEADING_DEG = None   # None = fitted from the data.  Or give the direction of the first leg in the room frame, in
                          # degrees counter-clockwise from room +X.  Phone held flat, top edge pointing along leg 1: 90.
STOP_MIN_S    = 0.8       # a still period counts as a STOP only if it lasts at least this long (s)
STOP_BRIDGE_S = 1.0       # tiny 'not still' blips shorter than this inside a stop are ignored (s)

# --- calibration table (used only when USE_CALIBRATION = True) ---
# Sensor model: reading = M @ true + bias.   We undo it:  true = M^-1 @ (reading - bias)
CALIBRATION_FILE = "/content/drive/MyDrive/colab_upload/phone/calibration.json"   # written by the calibration cell
CALIBRATION = {                                                                    # used if CALIBRATION_FILE is None
    "acc_bias":   [0.0, 0.0, 0.0],                              # m/s^2
    "acc_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],            # diagonal = scale, off-diagonal = crooked axes
    "gyro_bias":  [0.0, 0.0, 0.0],                              # rad/s
}

# --- finding the still moment at the start ---
AUTO_FIND_STILL = False   # True: search for a still period by itself. False: use the start times given above.
MANUAL_T_START  = 2.0     # s, EuRoC only. V1_01: 2.0 s is calm (picked by looking at the IMU plots).
STILL_LEN       = 0.6     # s, length of the still window
MAX_GYRO_STD    = 0.01    # a window is "still" if gyro wiggle is below this (rad/s)
MAX_ACC_STD     = 0.15    # ... and accelerometer wiggle is below this (m/s^2)
CHECK_AFTER     = 10.0    # s, tilt check is done at the last still moment later than this

# --- simple filters ---
KP = 0.5                  # Mahony: how hard the accelerometer pulls tilt back to "up"
KI = 0.0                  # Mahony: how fast it learns leftover gyro bias (0 = off)
ZUPT_GYRO = 0.05          # ZUPT: "still" if turn rate below this (rad/s)
ZUPT_ACC  = 0.3           # ZUPT: ... and accel length within this of gravity (m/s^2)
ZUPT_TIME = 0.2           # ZUPT: must stay quiet this long (s) before velocity is forced to zero

SHOW_SIMPLE_PLOT = True   # the easy plot: only the raw run and the final estimate
SHOW_FULL_PLOTS  = True   # the detailed plots: gyro + accel traces, all runs top-down / height / distance, 3D
FINAL_RUN = "auto"        # the run shown as the "final" estimate in the simple plot ("auto" = best run that uses no outside aid)
ZOOM_ON = "auto"          # which run decides the zoom of the detailed plots ("auto" picks a run that stays near the true path)

# --- ESKF: sensor noise from the dataset's sensor.yaml (continuous-time densities) ---
ACC_ND, ACC_RW = 2.0e-3, 3.0e-3          # m/s^2/sqrt(Hz), m/s^3/sqrt(Hz)
GYR_ND, GYR_RW = 1.6968e-4, 1.9393e-5    # rad/s/sqrt(Hz), rad/s^2/sqrt(Hz)
if SOURCE == "device":
    ACC_ND, GYR_ND = DEVICE_ACC_ND, DEVICE_GYR_ND    # device: use the values measured on the device
# Datasheet numbers describe clean white noise only; real error also has vibration and model error,
# so we inflate them. These factors are GUESSES (Cell 2 has an optional TUNE sweep).
ACC_NOISE_X = 20.0
GYR_NOISE_X = 10.0
BIAS_NOISE_X = 1.0

# --- ESKF: measurement noise and gates ---
SIG_GRAV  = 0.3      # gravity (tilt) update: large, because the device also accelerates
GRAV_GATE = 0.5      # use the gravity update only if | |acc| - g | is below this
GRAV_BIAS = 1.0      # 1 = gravity update may also adjust the accel bias, 0 = tilt only
GRAV_PERIOD = 0.5    # s between gravity updates (readings are not independent)
GRAV_ADAPT = 1.0     # gravity update distrusts itself more when the filter thinks it is accelerating (0 = off)
BA0_SIGMA = 0.5      # starting uncertainty of the accelerometer bias (m/s^2)
INIT_COUPLING = True     # tell the filter that the start-up tilt and accel bias are tied together (see eskf)
INIT_STILL_SIGMA = 0.05  # m/s^2: how tightly the start-up still window pins (tilt + accel bias) together
GRAV_ACC_GATE = 1.0  # only if the filter's own estimate of the real acceleration is below this (m/s^2)
SIG_ZUPT  = 0.02     # zero-velocity update
SIG_ZRATE = 0.01     # zero-rate (gyro) update
ZUPT_STD = 0.15      # FIX 1: still only if the accelerometer is also FLAT over the window (std, m/s^2).
                     # Was 0.05, which hand tremor can break; the printout below shows how often each test passes.
GATE_3 = 11.34       # chi-square 99% limit, 3 numbers  (outlier gate for position fixes)
GATE_1 = 6.63        # chi-square 99% limit, 1 number   (outlier gate for ranges)
REJECT_INFLATE = 1.2 # each rejected fix/range makes position, velocity and tilt this much less certain (1.0 = off)

# --- simulated UWB (EuRoC only): 4 fixed anchors (assumed layout around the room - edit freely), ranges only ---
UWB_ANCHORS = np.array([[-3.5, -3.5, 0.3], [3.5, -3.5, 2.7], [3.5, 4.0, 0.3], [-3.5, 4.0, 2.7]])
UWB_RATE, UWB_SIGMA, UWB_OUTLIER_P = 10.0, 0.15, 0.05   # Hz, m, chance that a reading is blocked (+0.3..1.5 m)

LOWPASS_HZ = 40.0    # optional causal low-pass on gyro and accel (second-order Butterworth); needs fs > 2*LOWPASS_HZ


# ---------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------
def _read(path):
    """EuRoC style csv: skip the '#timestamp...' header, return rows of text."""
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                rows.append(row)
    return rows


def load_euroc_imu(folder):
    """Returns t (s, 0 = first sample), gyro (rad/s), acc (m/s^2). Only the IMU file is opened."""
    imu_rows = _read(os.path.join(folder, "mav0", "imu0", "data.csv"))
    t_ns = np.array([int(r[0]) for r in imu_rows], dtype=np.int64)
    t = (t_ns - t_ns[0]) * 1e-9
    imu = np.array([[float(x) for x in r[1:7]] for r in imu_rows])
    return t, imu[:, 0:3], imu[:, 3:6]


def _read_columns(path, time_col, cols):
    """Device csv with a header row of names. Returns time array and N x 3 data."""
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        names = [h.strip() for h in r.fieldnames]
        r.fieldnames = names
        rows = list(r)
    need = [time_col] + list(cols)
    for c in need:
        if c not in names:
            raise KeyError("Column '%s' is not in %s. Columns found: %s" % (c, path, names))
    data = np.array([[float(row[c]) for c in need] for row in rows])
    return data[:, 0], data[:, 1:4]


def resolve_device_files(recording):
    """recording = a .zip or a folder. Returns the paths of the accelerometer and gyro csv files inside it."""
    if recording.lower().endswith(".zip"):
        import zipfile
        out = "/content/device_unzipped/" + os.path.splitext(os.path.basename(recording))[0]
        with zipfile.ZipFile(recording) as z:
            z.extractall(out)
        recording = out
    for root, dirs, files in os.walk(recording):
        if DEVICE_ACC_NAME in files and DEVICE_GYRO_NAME in files:
            return os.path.join(root, DEVICE_ACC_NAME), os.path.join(root, DEVICE_GYRO_NAME)
    raise FileNotFoundError("Did not find %s and %s inside %s" % (DEVICE_ACC_NAME, DEVICE_GYRO_NAME, recording))


def load_device(acc_file, gyro_file):
    """Reads accel + gyro, puts both on ONE clock at DEVICE_RATE_HZ (phones sample unevenly)."""
    ta, acc = _read_columns(acc_file, DEVICE_TIME_COL, DEVICE_ACC_COLS)
    tg, gyro = _read_columns(gyro_file, DEVICE_TIME_COL, DEVICE_GYRO_COLS)
    unit = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[DEVICE_TIME_UNIT]
    t0 = min(ta[0], tg[0])
    ta, tg = (ta - t0) * unit, (tg - t0) * unit
    t = np.arange(max(ta[0], tg[0]), min(ta[-1], tg[-1]), 1.0 / DEVICE_RATE_HZ)
    acc_u = np.column_stack([np.interp(t, ta, acc[:, i]) for i in range(3)]) * DEVICE_ACC_FACTOR
    gyro_u = np.column_stack([np.interp(t, tg, gyro[:, i]) for i in range(3)]) * DEVICE_GYRO_FACTOR
    return t - t[0], gyro_u, acc_u


def apply_calibration(gyro, acc, calib):
    """Undo the sensor model reading = M @ true + bias."""
    b_a = np.array(calib["acc_bias"], float)
    M = np.array(calib["acc_matrix"], float)
    b_g = np.array(calib["gyro_bias"], float)
    acc_c = (acc - b_a) @ np.linalg.inv(M).T
    gyro_c = gyro - b_g
    return gyro_c, acc_c


# ---------------------------------------------------------------------
# GROUND TRUTH - SCORING / DEV ONLY (EuRoC). Never called by the filters unless a dev switch is on.
# ---------------------------------------------------------------------
def load_truth(folder):
    """Returns t_gt (s, same clock as the IMU), pos_gt (m), q_gt (w, x, y, z)."""
    first = int(_read(os.path.join(folder, "mav0", "imu0", "data.csv"))[0][0])
    rows = _read(os.path.join(folder, "mav0", "state_groundtruth_estimate0", "data.csv"))
    t_gt = np.array([(int(r[0]) - first) * 1e-9 for r in rows])
    arr = np.array([[float(x) for x in r[1:8]] for r in rows])
    return t_gt, arr[:, 0:3], arr[:, 3:7]


def truth_start_orientation(folder, t_mid):
    """DEV ONLY. Start orientation from the ground-truth quaternion nearest to time t_mid."""
    t_gt, _, q_gt = load_truth(folder)
    q = q_gt[np.argmin(np.abs(t_gt - t_mid))]
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()      # SciPy wants x, y, z, w


# ---------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------
def R_from_accel(acc_still):
    """Start orientation from a still accelerometer average. Tilt is right; heading is arbitrary.
    Room Z = up, room X = the sensor's x axis flattened onto the floor, room Y = Z x X."""
    up = acc_still / np.linalg.norm(acc_still)
    row1 = np.array([1.0, 0.0, 0.0]) - up[0] * up
    row1 = row1 / np.linalg.norm(row1)
    row2 = np.cross(up, row1)
    return np.vstack([row1, row2, up])


def still_mask(gyro, acc, gyro_bias=(0.0, 0.0, 0.0), g=G, still_gyro=0.05, still_acc=0.3):
    """True for every sample where the sensor looks still (the same test ZUPT uses)."""
    turning = np.linalg.norm(np.asarray(gyro) - np.asarray(gyro_bias), axis=1)
    pushing = np.abs(np.linalg.norm(acc, axis=1) - g)
    return (turning < still_gyro) & (pushing < still_acc)


def window_mean(t, x, t_from, t_to):
    """Mean of x (N x k) over the samples with t_from <= t <= t_to."""
    m = (t >= t_from) & (t <= t_to)
    return x[m].mean(axis=0)


def find_still_windows(t, gyro, acc, g=G, length=0.6, step=0.1,
                       max_gyro_std=0.01, max_acc_std=0.15, acc_tol=0.3):
    """Slide a window along the recording; return the start time of every window that is still."""
    starts = []
    s = t[0]
    while s + length <= t[-1]:
        i0 = np.searchsorted(t, s)
        i1 = np.searchsorted(t, s + length)
        if i1 - i0 > 10:
            gy, ac = gyro[i0:i1], acc[i0:i1]
            quiet = (gy.std(axis=0).max() < max_gyro_std
                     and ac.std(axis=0).max() < max_acc_std
                     and abs(np.linalg.norm(ac.mean(axis=0)) - g) < acc_tol)
            if quiet:
                starts.append(s)
        s += step
    return np.array(starts)


# ---------------------------------------------------------------------
# STOPS and the two tests' ideal paths
# ---------------------------------------------------------------------
def _runs(mask):
    """List of (value, first_index, last_index) for every run of equal values in a True/False array."""
    out, i, n = [], 0, len(mask)
    while i < n:
        j = i
        while j + 1 < n and mask[j + 1] == mask[i]:
            j += 1
        out.append((bool(mask[i]), i, j))
        i = j + 1
    return out


def find_stops(t, still, min_stop_s=0.8, bridge_s=0.4):
    """Turn the sample-by-sample 'looks still' line into STOPS: periods that really are stops.
    1) short 'not still' blips inside a stop (hand tremor) are bridged, 2) still periods shorter than min_stop_s
    are dropped. Returns a list of (first_index, last_index)."""
    m = np.array(still, bool).copy()
    dt = float(np.median(np.diff(t)))
    for v, a, b in _runs(m)[1:-1]:
        if (not v) and (b - a + 1) * dt <= bridge_s:
            m[a:b + 1] = True
    return [(a, b) for v, a, b in _runs(m) if v and (b - a + 1) * dt >= min_stop_s]


def test_waypoints(test, turn="cw"):
    """Ideal stop positions (x, y in metres) in the TEST frame: start at (0,0), first leg along +x.
    Returns None for TEST = 'none'."""
    if test == "square":
        L = SQUARE_SIDE_M
        s = -1.0 if turn == "cw" else 1.0              # clockwise (right turns) goes to negative y
        return np.array([[0.0, 0.0], [L, 0.0], [L, s * L], [0.0, s * L], [0.0, 0.0]])
    if test == "slide":
        return np.array([[0.0 if i % 2 == 0 else SLIDE_LENGTH_M, 0.0] for i in range(SLIDE_PASSES + 1)])
    return None


def dedrift(t, a_w, stops, p_start, marks=None):
    """FIX 2: per-leg drift correction between stops.
    a_w   = acceleration in the room frame with gravity removed (N x 3), from any orientation estimate.
    stops = list of (first_index, last_index) from find_stops. Between two stops the device really moves; at both
            ends its velocity is really zero. The integrated velocity at the end of the leg is therefore pure error.
    Without marks: model the acceleration error as a constant e0 and remove it (makes the end velocity zero).
    With marks (dict stop_number -> known room-frame position): the arrival position is known too, so model the
    error as e0 + e1*tau/T and solve both constraints (end velocity = 0 AND arrival at the mark).
    Between stops the position is held still. Returns positions (N x 3)."""
    N = len(t)
    p_cur = np.array(p_start, float)
    pos = np.tile(p_cur, (N, 1))
    if len(stops) == 0:
        return pos
    pos[:stops[0][1] + 1] = p_cur
    for j in range(len(stops) - 1):
        a0, b0 = stops[j][1], stops[j + 1][0]            # leave stop j at a0, arrive at stop j+1 at b0
        if b0 <= a0:
            continue
        ts = t[a0:b0 + 1] - t[a0]
        T = ts[-1]
        dts = np.diff(ts)[:, None]
        a = a_w[a0:b0 + 1]
        v = np.vstack([np.zeros(3), np.cumsum(0.5 * (a[1:] + a[:-1]) * dts, axis=0)])   # v = 0 at the start
        p = np.vstack([np.zeros(3), np.cumsum(0.5 * (v[1:] + v[:-1]) * dts, axis=0)])
        if marks is not None and (j + 1) in marks:
            D = np.asarray(marks[j + 1], float) - p_cur                                  # where we must arrive
            A = np.array([[T, T / 2.0], [T ** 2 / 2.0, T ** 2 / 6.0]])
            e = np.linalg.solve(A, np.vstack([v[-1], p[-1] - D]))                        # 2 x 3
            e0, e1 = e[0], e[1]
        else:
            e0, e1 = v[-1] / T, np.zeros(3)
        tau = ts[:, None]
        pos[a0:b0 + 1] = p_cur + (p - 0.5 * e0 * tau ** 2 - e1 * tau ** 3 / (6.0 * T))
        p_cur = pos[b0].copy()
        pos[b0:stops[j + 1][1] + 1] = p_cur
    a0 = stops[-1][1]
    if a0 < N - 1:                                        # moving after the last stop, no stop to correct against
        ts = t[a0:] - t[a0]
        dts = np.diff(ts)[:, None]
        a = a_w[a0:]
        v = np.vstack([np.zeros(3), np.cumsum(0.5 * (a[1:] + a[:-1]) * dts, axis=0)])
        p = np.vstack([np.zeros(3), np.cumsum(0.5 * (v[1:] + v[:-1]) * dts, axis=0)])
        pos[a0:] = p_cur + p
    return pos


# ---------------------------------------------------------------------
# THE SIMPLE NAVIGATION LOOP (one sample in, one estimate out)
# ---------------------------------------------------------------------
def navigate(t, gyro, acc, R0, p0,
             gyro_bias=(0.0, 0.0, 0.0), acc_bias=(0.0, 0.0, 0.0), g=G,
             kp=0.0, ki=0.0, acc_gate=1.0,
             zupt=False, still_gyro=0.05, still_acc=0.3, still_time=0.2):
    g_vec = np.array([0.0, 0.0, g])
    N = len(t)
    R = np.array(R0, float)
    v = np.zeros(3)
    p = np.array(p0, float)
    gyro_bias = np.array(gyro_bias, float)
    acc = np.asarray(acc, float) - np.array(acc_bias, float)
    bias_est = np.zeros(3)
    quiet_since = None

    pos = np.zeros((N, 3)); Rs = np.zeros((N, 3, 3))
    pos[0] = p; Rs[0] = R

    for k in range(1, N):
        dt = t[k] - t[k - 1]
        a_norm = np.linalg.norm(acc[k])

        # orientation: gyro, plus a gentle pull toward "up" from the accelerometer (Mahony)
        err = np.zeros(3)
        if kp > 0 and abs(a_norm - g) < acc_gate:
            err = np.cross(acc[k] / a_norm, R[2, :])
            bias_est = bias_est - ki * err * dt
        w = gyro[k] - gyro_bias - bias_est + kp * err
        R = R @ Rot.from_rotvec(w * dt).as_matrix()

        # rotate push into the room, remove gravity, integrate twice (trapezoid)
        a_world = R @ acc[k] - g_vec
        v_new = v + a_world * dt
        p = p + 0.5 * (v + v_new) * dt
        v = v_new

        # ZUPT: if quiet for a while, velocity must be zero
        if zupt:
            quiet = (np.linalg.norm(gyro[k] - gyro_bias) < still_gyro) and (abs(a_norm - g) < still_acc)
            if quiet:
                if quiet_since is None:
                    quiet_since = t[k]
                if t[k] - quiet_since >= still_time:
                    v = np.zeros(3)
            else:
                quiet_since = None

        pos[k] = p; Rs[k] = R
    return pos, Rs


# ---------------------------------------------------------------------
# ESKF: maths helpers
# ---------------------------------------------------------------------
def skew(v):
    """Matrix that does the cross product: skew(a) @ b = a x b."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w):
    """Rotation matrix for a small turn vector w (rad). Rodrigues formula."""
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def lowpass(x, fs, fc, order=2):
    """Causal low-pass (usable in real time). Starts from the first sample, not from zero."""
    b, a = butter(order, fc / (0.5 * fs))
    zi = lfilter_zi(b, a)[:, None] * x[0][None, :]
    return lfilter(b, a, x, axis=0, zi=zi)[0]


def rolling_std(x, n):
    """Standard deviation of each column of x over the last n samples (vectorised)."""
    z = np.zeros((1, x.shape[1]))
    c1, c2 = np.cumsum(np.vstack([z, x]), axis=0), np.cumsum(np.vstack([z, x ** 2]), axis=0)
    idx = np.arange(len(x)); lo = np.maximum(idx - n + 1, 0); cnt = (idx - lo + 1)[:, None]
    var = (c2[idx + 1] - c2[lo]) / cnt - ((c1[idx + 1] - c1[lo]) / cnt) ** 2
    return np.sqrt(np.maximum(var, 0.0))


def _upd(P, H, y, Rm, gate):
    """One Kalman update (Joseph form). Returns (state correction or None if rejected, new P).
    y = measurement minus prediction. gate = chi-square limit or None (no gating)."""
    S = H @ P @ H.T + Rm
    nis = float(y @ np.linalg.solve(S, y))            # how surprising is this measurement?
    if gate is not None and nis > gate:
        return None, P
    K = np.linalg.solve(S, H @ P).T
    IKH = np.eye(P.shape[0]) - K @ H
    return K @ y, IKH @ P @ IKH.T + K @ Rm @ K.T


def eskf(t, gyro, acc, R0, p0, g, bg0, fixes=None, ranges=None,
         use_grav=True, use_still=True, use_gate=True, acc_x=None, gyr_x=None, bias_x=None, acc_for_still=None,
         init_coupling=None, ba0=None):
    """Error-state Kalman filter. State errors (15): position, velocity, tilt/heading, accel bias, gyro bias.
    fixes = (times, positions N x 3, sigma);  ranges = (times, N x 4 distances, sigma)."""
    init_coupling = INIT_COUPLING if init_coupling is None else init_coupling
    acc_x = ACC_NOISE_X if acc_x is None else acc_x
    gyr_x = GYR_NOISE_X if gyr_x is None else gyr_x
    bias_x = BIAS_NOISE_X if bias_x is None else bias_x
    N = len(t)
    I3, I15 = np.eye(3), np.eye(15)
    gv = np.array([0.0, 0.0, g])
    p, v, R = np.array(p0, float), np.zeros(3), np.array(R0, float)
    ba, bg = np.zeros(3), np.array(bg0, float)
    P = np.diag([0.05 ** 2] * 3 + [0.02 ** 2] * 3 + [0.05 ** 2] * 3 + [BA0_SIGMA ** 2] * 3 + [0.005 ** 2] * 3)
    qd = np.concatenate([[1e-8] * 3, [(ACC_ND * acc_x) ** 2] * 3, [(GYR_ND * gyr_x) ** 2] * 3,
                         [(ACC_RW * bias_x) ** 2] * 3, [(GYR_RW * bias_x) ** 2] * 3])
    if ba0 is not None:
        # DEV ONLY (oracle start): tilt and accel bias come from truth, so both start tight.
        ba = np.array(ba0, float)
        P[6:9, 6:9] = 0.01 ** 2 * I3
        P[9:12, 9:12] = 0.05 ** 2 * I3
    elif init_coupling:
        # R0 was built FROM the still accelerometer, so a (horizontal) tilt error and an accel-bias error cancel
        # exactly there: the filter cannot tell them apart. Say so with one virtual 'gravity matches' measurement
        # (innovation 0): P gets the right tilt <-> bias correlation, the estimate does not move.
        un = (R.T @ gv) / g
        a1 = np.cross(un, [1.0, 0.0, 0.0] if abs(un[0]) < 0.9 else [0.0, 1.0, 0.0]); a1 /= np.linalg.norm(a1)
        B = np.vstack([a1, np.cross(un, a1)])                  # 2 x 3, plane perpendicular to 'up'
        H0 = np.zeros((2, 15)); H0[:, 6:9] = B @ skew(R.T @ gv); H0[:, 9:12] = B
        _, P = _upd(P, H0, np.zeros(2), INIT_STILL_SIGMA ** 2 * np.eye(2), None)
    diag = np.diag_indices(15)
    acc_flat = rolling_std(np.asarray(acc if acc_for_still is None else acc_for_still, float),
                           max(int(round(ZUPT_TIME / np.median(np.diff(t)))), 3)).max(axis=1)

    pos, Rs = np.zeros((N, 3)), np.zeros((N, 3, 3))
    pstd, BA, BG = np.zeros(N), np.zeros((N, 3)), np.zeros((N, 3))
    pos[0], Rs[0], pstd[0], BA[0], BG[0] = p, R, np.sqrt(np.trace(P[0:3, 0:3])), ba, bg
    stats = dict(rejected=0, accepted=0, still=0)
    quiet_since, ifix, irng, rej_run, last_grav = None, 0, 0, 0, -1e9

    def apply(H, y, Rm, gate, inflate=False):
        nonlocal p, v, R, ba, bg, P
        dx, P = _upd(P, H, y, Rm, gate)
        if dx is None:
            stats["rejected"] += 1
            if inflate:                                    # filter was too sure of itself: widen it
                s = np.ones(15); s[0:9] = np.sqrt(REJECT_INFLATE)
                P = P * np.outer(s, s)
            return False
        p, v = p + dx[0:3], v + dx[3:6]
        R = R @ so3_exp(dx[6:9])
        ba, bg = ba + dx[9:12], bg + dx[12:15]
        stats["accepted"] += 1
        return True

    for k in range(1, N):
        dt = t[k] - t[k - 1]
        w, f = gyro[k] - bg, acc[k] - ba

        # ---- predict: nominal state, then the error covariance ----
        a_w = R @ f - gv
        F = I15.copy()
        F[0:3, 3:6] = I3 * dt
        F[3:6, 6:9] = -(R @ skew(f)) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = so3_exp(-w * dt)
        F[6:9, 12:15] = -I3 * dt
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        R = R @ so3_exp(w * dt)
        P = F @ P @ F.T
        P[diag] += qd * dt
        if k % 200 == 0:                                   # keep R a true rotation matrix
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt

        # ---- update 1: gravity says which way is up (fixes tilt, learns accel bias) ----
        if (use_grav and t[k] - last_grav >= GRAV_PERIOD and abs(np.linalg.norm(acc[k]) - g) < GRAV_GATE
                and np.linalg.norm(a_w) < GRAV_ACC_GATE):
            last_grav = t[k]
            u = R.T @ gv
            H = np.zeros((3, 15)); H[:, 6:9] = skew(u); H[:, 9:12] = I3 * GRAV_BIAS
            apply(H, acc[k] - (u + ba), (SIG_GRAV ** 2 + (GRAV_ADAPT * np.linalg.norm(a_w)) ** 2) * I3,
                  GATE_3 if use_gate else None)

        # ---- update 2: when still, velocity = 0 and gyro = its bias ----
        if use_still:
            quiet = np.linalg.norm(gyro[k] - bg) < ZUPT_GYRO and abs(np.linalg.norm(acc[k] - ba) - g) < ZUPT_ACC \
                    and acc_flat[k] < ZUPT_STD
            if quiet:
                if quiet_since is None:
                    quiet_since = t[k]
                if t[k] - quiet_since >= ZUPT_TIME:
                    H = np.zeros((3, 15)); H[:, 3:6] = I3
                    apply(H, -v, SIG_ZUPT ** 2 * I3, None)
                    stats["still"] += 1
                    H = np.zeros((3, 15)); H[:, 12:15] = I3
                    apply(H, gyro[k] - bg, SIG_ZRATE ** 2 * I3, None)
            else:
                quiet_since = None

        # ---- update 3: position fix (anchor / tag / floor mark) ----
        if fixes is not None and ifix < len(fixes[0]) and t[k] >= fixes[0][ifix]:
            H = np.zeros((3, 15)); H[:, 0:3] = I3
            ok = apply(H, fixes[1][ifix] - p, fixes[2] ** 2 * I3, GATE_3 if use_gate else None, inflate=True)
            rej_run = 0 if ok else rej_run + 1
            if rej_run >= 3:                               # filter may be lost: re-acquire from the fix
                apply(H, fixes[1][ifix] - p, fixes[2] ** 2 * I3, None)
                rej_run = 0
            ifix += 1

        # ---- update 4: UWB ranges (distance to each fixed anchor) ----
        if ranges is not None and irng < len(ranges[0]) and t[k] >= ranges[0][irng]:
            for i in range(len(UWB_ANCHORS)):
                d = p - UWB_ANCHORS[i]
                r = np.linalg.norm(d)
                H = np.zeros((1, 15)); H[0, 0:3] = d / r
                apply(H, np.array([ranges[1][irng, i] - r]), np.array([[ranges[2] ** 2]]),
                      GATE_1 if use_gate else None, inflate=True)
            irng += 1

        pos[k], Rs[k] = p, R
        pstd[k], BA[k], BG[k] = np.sqrt(np.trace(P[0:3, 0:3])), ba, bg
    return pos, Rs, dict(pstd=pstd, ba=BA, bg=BG, **stats)


# ---------------------------------------------------------------------
# CHECKS THAT NEED NO GROUND TRUTH
# ---------------------------------------------------------------------
def path_length(pos):
    """Running total of the distance travelled along a path."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])


def tilt_check_deg(R, acc_still_mean):
    """Angle between the 'up' the accelerometer sees (at a still moment) and the 'up' the filter believes."""
    up_meas = acc_still_mean / np.linalg.norm(acc_still_mean)
    up_est = R[2, :]
    return float(np.degrees(np.arccos(np.clip(up_meas @ up_est, -1.0, 1.0))))


# ---------------------------------------------------------------------
# PLOTS (no ground truth)
# ---------------------------------------------------------------------
def plot_raw_sensors(t, data, title, unit, names=("x", "y", "z")):
    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, c in enumerate("bgr"):
        axs[i].plot(t, data[:, i], color=c, label=f"{title} {names[i]}")
        axs[i].set_ylabel(unit); axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("Time (s)"); fig.suptitle(title)
    return fig


def plot_paths(t, runs, zoom_on):
    ref = runs[zoom_on][0]
    lo, hi = ref.min(0), ref.max(0)
    m = 0.5 * max((hi - lo)[:2].max(), 1.0)
    mz = 0.5 * max(hi[2] - lo[2], 1.0)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    for name, (pos, _) in runs.items():
        axs[0].plot(pos[:, 0], pos[:, 1], label=name)
        axs[1].plot(t, pos[:, 2], label=name)
        d = np.maximum(np.linalg.norm(pos - pos[0], axis=1), 1e-3)
        axs[2].plot(t, d, label=name)
    axs[0].set_xlim(lo[0] - m, hi[0] + m); axs[0].set_ylim(lo[1] - m, hi[1] + m); axs[0].set_aspect("equal")
    axs[0].set_title("Top-down path (X-Y), zoomed on '%s'" % zoom_on)
    axs[0].set_xlabel("X (m)"); axs[0].set_ylabel("Y (m)"); axs[0].legend(fontsize=7)
    axs[1].set_ylim(lo[2] - mz, hi[2] + mz)
    axs[1].set_title("Height Z vs time"); axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("Z (m)")
    axs[2].set_title("Distance from the start vs time"); axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("distance (m)"); axs[2].set_yscale("log")
    plt.tight_layout()
    return fig


def plot_3d(runs, zoom_on, triad_run=None, every=300):
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(projection="3d")
    ref = runs[zoom_on][0]
    lo, hi = ref.min(0), ref.max(0)
    margin = 0.3 * max(np.ptp(ref, axis=0).max(), 0.5)
    lo, hi = lo - margin, hi + margin

    def inside(pos):
        ok = np.all((pos >= lo) & (pos <= hi), axis=1)
        return np.where(ok[:, None], pos, np.nan)

    for name, (pos, _) in runs.items():
        ax.plot(*inside(pos).T, label=name)

    L = 0.12 * (hi - lo).max()
    if triad_run is not None:
        pos, Rs = runs[triad_run]
        for k in range(0, len(pos), every):
            if np.all((pos[k] >= lo) & (pos[k] <= hi)):
                for i, c in enumerate("rgb"):
                    ax.quiver(*pos[k], *(Rs[k][:, i] * L), color=c, linewidth=1.2)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-6)))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(fontsize=7)
    return fig


def plot_summary(t, runs, final):
    """ONE clean figure: the raw run and the final estimate only."""
    pf, pr = runs[final][0], runs["raw"][0]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    lo, hi = pf[:, :2].min(0), pf[:, :2].max(0)
    m = 0.5 * max((hi - lo).max(), 1.0)
    axs[0].plot(pr[:, 0], pr[:, 1], color="tab:red", lw=1.4,
                label="raw (ends %.0f m from the start)" % np.linalg.norm(pr[-1] - pr[0]))
    axs[0].plot(pf[:, 0], pf[:, 1], color="tab:blue", lw=1.8,
                label="%s (ends %.2f m from the start)" % (final, np.linalg.norm(pf[-1] - pf[0])))
    axs[0].plot(0, 0, "ko", ms=8, label="start")
    axs[0].set_xlim(lo[0] - m, hi[0] + m); axs[0].set_ylim(lo[1] - m, hi[1] + m); axs[0].set_aspect("equal")
    axs[0].set_title("Top-down path (zoomed on the final estimate)")
    axs[0].set_xlabel("X (m)"); axs[0].set_ylabel("Y (m)"); axs[0].legend(fontsize=8)
    for pos, c, n in ((pr, "tab:red", "raw"), (pf, "tab:blue", final)):
        axs[1].plot(t, np.maximum(np.linalg.norm(pos - pos[0], axis=1), 1e-3), color=c, label=n)
    axs[1].set_yscale("log"); axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("distance from the start (m)")
    axs[1].set_title("Distance from the start vs time"); axs[1].legend(fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------
# MAIN: load -> (calibrate) -> still moment -> start condition -> runs -> report -> plots
# ---------------------------------------------------------------------
def main():
    # 1. load
    test = TEST if SOURCE == "device" else "none"
    if SOURCE == "euroc":
        t_all, gyro_all, acc_all = load_euroc_imu(EUROC_FOLDER)
    elif SOURCE == "device":
        if TEST not in DEVICE_RECORDINGS:
            raise ValueError("TEST must be one of %s" % list(DEVICE_RECORDINGS))
        print("TEST = %s | recording: %s" % (TEST, DEVICE_RECORDINGS[TEST]))
        t_all, gyro_all, acc_all = load_device(*resolve_device_files(DEVICE_RECORDINGS[TEST]))
    else:
        raise ValueError("SOURCE must be 'euroc' or 'device'")
    print("source: %s | IMU samples: %d | duration %.1f s" % (SOURCE, len(t_all), t_all[-1]))

    # 2. optional calibration table (device only: the table describes the device, not the drone)
    if USE_CALIBRATION and SOURCE == "device":
        calib = json.load(open(CALIBRATION_FILE)) if CALIBRATION_FILE else CALIBRATION
        gyro_all, acc_all = apply_calibration(gyro_all, acc_all, calib)
        print("calibration: ON  ->", calib)
    else:
        print("calibration: OFF (raw sensor data)")

    # 3. still moment
    starts = find_still_windows(t_all, gyro_all, acc_all, length=STILL_LEN,
                                max_gyro_std=MAX_GYRO_STD, max_acc_std=MAX_ACC_STD)
    if AUTO_FIND_STILL:
        if len(starts) == 0:
            raise RuntimeError("No still window found. Loosen MAX_GYRO_STD / MAX_ACC_STD, "
                               "or set AUTO_FIND_STILL = False and choose the start time.")
        t_start = float(starts[0])
    else:
        t_start = MANUAL_T_START if SOURCE == "euroc" else DEVICE_T_START[TEST]
    still_end = t_start + STILL_LEN
    print("still window used      : %.2f .. %.2f s" % (t_start, still_end))

    keep = t_all >= t_start
    t, gyro, acc = t_all[keep], gyro_all[keep], acc_all[keep]
    print("navigating from %.2f s to %.2f s (%d samples)" % (t[0], t[-1], len(t)))

    # 4. start condition, learned from the still window
    gyro_bias = window_mean(t, gyro, t_start, still_end)
    acc_still = window_mean(t, acc, t_start, still_end)
    g_est = float(np.linalg.norm(acc_still))
    R0 = R_from_accel(acc_still)
    p0 = np.zeros(3)
    acc_bias = np.zeros(3)

    if USE_TRUTH_FOR_START:
        if SOURCE != "euroc":
            raise ValueError("USE_TRUTH_FOR_START only works with SOURCE = 'euroc' (a device has no truth)")
        R0 = truth_start_orientation(EUROC_FOLDER, 0.5 * (t_start + still_end))
        acc_bias = acc_still - R0.T @ np.array([0.0, 0.0, g_est])
        print("TRUTH MODE ON (dev only). accel bias taken from truth:", np.round(acc_bias, 3))

    w_ = (t >= t_start) & (t <= still_end)
    if gyro[w_].std(axis=0).max() > MAX_GYRO_STD or acc[w_].std(axis=0).max() > MAX_ACC_STD:
        print("WARNING: the start window %.1f-%.1f s is NOT still (gyro std %.3f, accel std %.3f). Pick another start time."
              % (t_start, still_end, gyro[w_].std(axis=0).max(), acc[w_].std(axis=0).max()))
    print("gyro bias estimate     :", np.round(gyro_bias, 4))
    print("still accel measured   :", np.round(acc_still, 3), "| length (used as g) %.3f" % g_est)

    sm = still_mask(gyro, acc, gyro_bias, g=g_est, still_gyro=ZUPT_GYRO, still_acc=ZUPT_ACC)
    print("ZUPT test fires on %.0f%% of ALL navigated samples" % (100 * sm.mean()))

    # 5. runs
    kw = dict(gyro_bias=gyro_bias, acc_bias=acc_bias, g=g_est)
    zk = dict(zupt=True, still_gyro=ZUPT_GYRO, still_acc=ZUPT_ACC, still_time=ZUPT_TIME)
    runs = {}
    print("running: raw ...")
    runs["raw"] = navigate(t, gyro, acc, R0, p0)
    print("running: bias removed ...")
    runs["bias removed"] = navigate(t, gyro, acc, R0, p0, **kw)
    print("running: bias + Mahony ...")
    runs["bias + Mahony"] = navigate(t, gyro, acc, R0, p0, **kw, kp=KP, ki=KI)
    print("running: bias + ZUPT ...")
    runs["bias + ZUPT"] = navigate(t, gyro, acc, R0, p0, **kw, **zk)
    print("running: all combined ...")
    runs["all combined"] = navigate(t, gyro, acc, R0, p0, **kw, kp=KP, ki=KI, **zk)
    eskf_info = None
    if RUN_ESKF:
        print("running: ESKF (no aiding) ...")
        pos_e, Rs_e, eskf_info = eskf(t, gyro, acc, R0, p0, g_est, gyro_bias,
                                     ba0=acc_bias if USE_TRUTH_FOR_START else None)
        runs["ESKF (no aiding)"] = (pos_e, Rs_e)

    # 5b. ESKF stop-test diagnostic: which of its three conditions blocks the zero-velocity updates?
    if eskf_info is not None:
        n_flat = max(int(round(ZUPT_TIME / np.median(np.diff(t)))), 3)
        c_gyro = np.linalg.norm(gyro - eskf_info["bg"], axis=1) < ZUPT_GYRO
        c_len = np.abs(np.linalg.norm(acc - eskf_info["ba"], axis=1) - g_est) < ZUPT_ACC
        c_flat = rolling_std(acc, n_flat).max(axis=1) < ZUPT_STD
        pc = lambda m: "%3.0f%%" % (100 * m.mean())
        print("ESKF stop test passes: gyro quiet %s | length near g %s | accel flat (ZUPT_STD=%.2f) %s | all three %s"
              % (pc(c_gyro), pc(c_len), ZUPT_STD, pc(c_flat), pc(c_gyro & c_len & c_flat)))

    # 5c. FIX 2 (device only): per-leg drift correction between the stops
    stops = []
    if SOURCE == "device":
        stops = find_stops(t, sm, STOP_MIN_S, STOP_BRIDGE_S)
        print("stops found (%d): %s" % (len(stops), ", ".join("%.1f-%.1f s" % (t[a], t[b]) for a, b in stops)))
        if len(stops) >= 2:
            g_vec = np.array([0.0, 0.0, g_est])
            print("running: leg de-drift (Mahony) ...")
            Rs_m = runs["bias + Mahony"][1]
            a_m = np.einsum("kij,kj->ki", Rs_m, acc) - g_vec
            runs["leg de-drift (Mahony)"] = (dedrift(t, a_m, stops, p0), Rs_m)
            if eskf_info is not None:
                print("running: leg de-drift (ESKF) ...")
                Rs_k = runs["ESKF (no aiding)"][1]
                a_k = np.einsum("kij,kj->ki", Rs_k, acc - eskf_info["ba"]) - g_vec
                runs["leg de-drift (ESKF)"] = (dedrift(t, a_k, stops, p0), Rs_k)
        else:
            print("fewer than 2 stops found, so the per-leg correction is skipped "
                  "(lower STOP_MIN_S, or check the still test above)")

    # 6. a later still moment, for a tilt check
    late = starts[starts >= t_start + CHECK_AFTER] if len(starts) else np.array([])
    i_chk, acc_chk = None, None
    if len(late):
        w0 = float(late[-1])
        mw = (t >= w0) & (t <= w0 + STILL_LEN)
        acc_chk = acc[mw].mean(axis=0)
        i_chk = int(np.argmax(mw)) + int(mw.sum()) // 2
        print("\ntilt check done at t = %.1f s (a later still moment)" % t[i_chk])
    else:
        print("\nno later still moment found, so no tilt check")

    # 7. report (no truth: consistency checks, NOT true errors)
    print()
    print("%-24s %14s %14s %16s %16s" % ("run", "path len (m)", "max dist (m)", "end-to-start (m)", "tilt check (deg)"))
    for name, (pos, Rs) in runs.items():
        plen = path_length(pos)[-1]
        dmax = np.linalg.norm(pos - pos[0], axis=1).max()
        dend = np.linalg.norm(pos[-1] - pos[0])
        tilt = tilt_check_deg(Rs[i_chk], acc_chk) if i_chk is not None else float("nan")
        print("%-24s %14.2f %14.2f %16.2f %16.2f" % (name, plen, dmax, dend, tilt))
    if eskf_info is not None:
        print("ESKF (no aiding): %d still-updates applied, %d measurements rejected by the gate" %
              (eskf_info["still"], eskf_info["rejected"]))
    print("No truth here, so these are not real errors. Run Cell 2 for the real scoring.")

    # 8. plots (no ground truth)
    final = FINAL_RUN
    if final == "auto" or final not in runs:
        final = next((n for n in ("leg de-drift (ESKF)", "ESKF (no aiding)", "all combined") if n in runs), list(runs)[-1])
    if SHOW_SIMPLE_PLOT:
        plot_summary(t, runs, final)
    if SHOW_FULL_PLOTS:
        zoom = ZOOM_ON
        if zoom == "auto" or zoom not in runs:
            zoom = next((n for n in ("leg de-drift (Mahony)", "all combined", "bias removed") if n in runs), list(runs)[0])
            if SOURCE == "euroc":
                zoom = "bias removed"
        plot_raw_sensors(t, gyro, "Gyroscope", "rad/s")
        plot_raw_sensors(t, acc, "Accelerometer", "m/s^2")
        plot_paths(t, runs, zoom)
        plot_3d(runs, zoom, triad_run=zoom)
    plt.show()

    return {"t": t, "gyro": gyro, "acc": acc, "runs": runs, "source": SOURCE, "test": test,
            "t_start": t_start, "still_end": still_end, "acc_still": acc_still, "R0": R0, "p0": p0,
            "gyro_bias": gyro_bias, "g_est": g_est, "eskf_info": eskf_info, "stops": stops, "still": sm}


results = main()
