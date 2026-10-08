"""
Biomechanics manuscript, analysis v2 (2026-10-07).

Re-runs the kinematic-asymmetry, BMI and sex analyses with three corrections
to the v1 scripts (project1_bmi.py, project2_sex.py, project4_hierarchy.py):

1. Force-plate gait events are debounced. v1 took every rising edge of
   Fz > 20 N, which also fires on 1-7 ms noise blips, so many v1 "cycles"
   were stride fragments (worst at 0.75 and 1.0 m/s).
2. Kinematic events come from the IMU itself (Zeni et al. 2008 foot-pelvis
   method). The IMU and force-plate files carry no shared sync signal
   (all Sync/Aux/Timestamp columns are zero), so mapping FP sample indices
   onto IMU frames is not valid.
3. KA(t) compares each leg over its OWN gait cycle:
   KA(t) = mean right cycle (right HS -> right HS) - mean left cycle
   (left HS -> left HS). v1 subtracted the two legs inside the same left-HS
   window, i.e. two waveforms ~50 % of a cycle out of phase.

Also: GRF normalised to body weight; ICC(C,1) and ICC(A,1) computed directly
(McGraw & Wong 1996) and for three KA summaries (mean |KA|, static offset,
dynamic RMS); diagnostic checks of FP/IMU/EMG alignment.

v3 (2026-10-07): adds force-plate baseline correction (median outside contacts),
as in utils_v4.py of the EEG paper. Outputs go to ../results_v3/.
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats
import spm1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(PROJECT, "Dataset")
OUT = os.path.abspath(os.path.join(HERE, "..", "results_v3"))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

SPEEDS = [0.5, 0.75, 1.0]
SUBJECTS = list(range(1, 60))
FS_FP, FS_IMU = 1000, 60
G = 9.81
JOINTS = {
    "hip_flex": ("Hip Flexion/Extension", "Hip flexion/extension"),
    "hip_abd": ("Hip Abduction/Adduction", "Hip abduction/adduction"),
    "knee_flex": ("Knee Flexion/Extension", "Knee flexion/extension"),
    "ankle": ("Ankle Dorsiflexion/Plantarflexion", "Ankle dorsi/plantarflexion"),
}
FP_COLS = ["1:Timestamp", "1:Sync", "1:Aux", "1:Fx", "1:Fy", "1:Fz", "1:Mx", "1:My", "1:Mz", "1:COPx", "1:COPy",
           "2:Timestamp", "2:Sync", "2:Aux", "2:Fx", "2:Fy", "2:Fz", "2:Mx", "2:My", "2:Mz", "2:COPx", "2:COPy"]


def sp_str(sp):
    return "1" if sp == 1.0 else str(sp)


# ── loaders ────────────────────────────────────────────────────────────────
def load_meta():
    m = pd.read_csv(os.path.join(DATA, "subject-info.csv"))
    m = m[m["Subject"].astype(str).str.match(r"^S\d+$")].copy()
    m["sid"] = m["Subject"].str[1:].astype(int)
    m["sex"] = m["GENDER"].str.strip().str.upper().str[0]
    m["BMI"] = m["BMI"].astype(float)
    m["mass"] = m["WEIGHT(kg)"].astype(float)
    m["Age"] = m["Age"].astype(float)
    m["bmi_group"] = pd.cut(m["BMI"], [0, 25, 30, 100], right=False, labels=["Normal", "Overweight", "Obese"])
    return m.set_index("sid")


def load_fp(sid, sp):
    p = os.path.join(DATA, "FP_Data", f"FP_S{sid}_{sp_str(sp)}.csv")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        first = fh.readline()
    if "Fz" in first:
        d = pd.read_csv(p, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
    else:
        d = pd.read_csv(p, header=None, low_memory=False)
        d.columns = FP_COLS[: d.shape[1]]
    return d


def load_imu(sid, sp, name):
    p = os.path.join(DATA, "IMU_Data", f"IMU_S{sid}_{sp_str(sp)}", f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_emg(sid, sp):
    p = os.path.join(DATA, "EMG_Data", f"EMG_S{sid}_{sp_str(sp)}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


# ── signal helpers ─────────────────────────────────────────────────────────
def tn(x, n=101):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def lowpass(x, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x)


def fp_contacts(fz, thr=50.0, min_dur=0.2, min_peak=200.0, merge_gap=0.05, warmup_s=2.0):
    """Debounced stance runs on one plate. Returns list of (on, off, peak)."""
    st = (fz > thr).astype(np.int8)
    d = np.diff(np.r_[0, st, 0])
    on, off = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = []
    for a, b in zip(on, off):
        if runs and a - runs[-1][1] < merge_gap * FS_FP:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    out = []
    for a, b in runs:
        if a < warmup_s * FS_FP:
            continue
        if (b - a) >= min_dur * FS_FP and fz[a:b].max() >= min_peak:
            out.append((a, b, float(fz[a:b].max())))
    return out


def baseline_correct(x, contacts, pad=50):
    """Subtract the median of the signal outside contacts (+/- pad samples)."""
    m = np.ones(len(x), bool)
    for a, b, *_ in contacts:
        m[max(a - pad, 0):b + pad] = False
    base = float(np.median(x[m])) if m.sum() > 100 else 0.0
    return x - base, base


def clean_strides(hs, fs, lo=0.6, hi=2.5, tol=0.3):
    hs = np.asarray(hs)
    if len(hs) < 3:
        return []
    dur = np.diff(hs) / fs
    med = np.median(dur[(dur >= lo) & (dur <= hi)]) if np.any((dur >= lo) & (dur <= hi)) else np.nan
    keep = []
    for i, dd in enumerate(dur):
        if lo <= dd <= hi and abs(dd - med) <= tol * med:
            keep.append((int(hs[i]), int(hs[i + 1])))
    return keep


def imu_heel_strikes(pos, side):
    """Zeni et al. (2008): HS = max anterior foot position relative to pelvis."""
    fx = pos[f"{side} Foot x"].values - pos["Pelvis x"].values
    fy = pos[f"{side} Foot y"].values - pos["Pelvis y"].values
    xy = np.c_[fx - fx.mean(), fy - fy.mean()]
    u = np.linalg.svd(xy, full_matrices=False)[2][0]
    p = lowpass(xy @ u, 6.0, FS_IMU)
    # stride period from autocorrelation
    ac = np.correlate(p - p.mean(), p - p.mean(), "full")[len(p) - 1:]
    lag_lo, lag_hi = int(0.6 * FS_IMU), int(2.5 * FS_IMU)
    per = lag_lo + int(np.argmax(ac[lag_lo:lag_hi]))
    rng = np.percentile(p, 95) - np.percentile(p, 5)
    best = None
    for sign in (1, -1):
        mx, _ = find_peaks(sign * p, distance=int(0.6 * per), prominence=0.3 * rng)
        mn, _ = find_peaks(-sign * p, distance=int(0.6 * per), prominence=0.3 * rng)
        if len(mx) < 3 or len(mn) < 3:
            continue
        # stance = HS (max) -> TO (next min); should be ~55-70 % of stride
        fr = []
        for i in range(len(mx) - 1):
            nxt = mn[(mn > mx[i]) & (mn < mx[i + 1])]
            if len(nxt):
                fr.append((nxt[0] - mx[i]) / (mx[i + 1] - mx[i]))
        f = np.median(fr) if fr else 0
        if best is None or abs(f - 0.62) < abs(best[1] - 0.62):
            best = (mx, f)
    if best is None:
        return np.array([]), np.nan, np.nan
    return best[0], per / FS_IMU, best[1]


# ── ICC (McGraw & Wong 1996), formulas as implemented in pingouin ─────────
def icc_all(Y):
    """Y: n subjects x k conditions. Returns dict with ICC(C,1), ICC(A,1) and 95 % CIs."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    gm = Y.mean()
    ssr = k * ((Y.mean(1) - gm) ** 2).sum()
    ssc = n * ((Y.mean(0) - gm) ** 2).sum()
    sst = ((Y - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc = ssr / (n - 1), ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    c1 = (msr - mse) / (msr + (k - 1) * mse)
    a1 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    F = msr / mse
    df1, df2 = n - 1, (n - 1) * (k - 1)
    fl = F / stats.f.ppf(0.975, df1, df2)
    fu = F * stats.f.ppf(0.975, df2, df1)
    c_ci = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
    a = k * a1 / (n * (1 - a1))
    b = 1 + k * a1 * (n - 1) / (n * (1 - a1))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    FL = stats.f.ppf(0.975, n - 1, v)
    FU = stats.f.ppf(0.975, v, n - 1)
    lo = n * (msr - FL * mse) / (FL * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi = n * (FU * msr - mse) / (k * msc + (k * n - k - n) * mse + n * FU * msr)
    return dict(n=n, ICC_C1=c1, C1_lo=c_ci[0], C1_hi=c_ci[1], ICC_A1=a1, A1_lo=lo, A1_hi=hi)


def clusters_of(spmi):
    out = []
    for c in spmi.clusters:
        e0, e1 = float(c.endpoints[0]), float(c.endpoints[1])
        out.append(f"{e0:.0f}-{e1:.0f}% (p={c.P if hasattr(c, 'P') else c.p:.3g})")
    return "; ".join(out) if out else "none"


# ═══════════════════════════════════════════════════════════════════════════
def main():
    meta = load_meta()
    trials, diag_rows, emg_rows = [], [], []
    kin = {}      # (sid, sp) -> dict joint -> {"R": mean cycle, "L": mean cycle, "KA_v1": ...}
    fpw = {}      # (sid, sp) -> dict channel -> mean cycle (plate 1, BW-normalised)
    fai = {}      # (sid, sp) -> FAI peak Fz (%)
    for sid in SUBJECTS:
        mass = meta.loc[sid, "mass"]
        for sp in SPEEDS:
            rec = dict(sid=sid, speed=sp)
            fp = load_fp(sid, sp)
            pos = load_imu(sid, sp, "Segment_Position")
            ang = load_imu(sid, sp, "Joint_Angles_ZXY")
            # ---------------- force plate ----------------
            hs_fp = {}
            if fp is not None:
                rec["fp_s"] = len(fp) / FS_FP
                rec["fp_sync_nonzero"] = int(sum((fp[c].abs() > 0).sum() for c in fp.columns
                                                 if c.endswith(("Sync", "Aux", "Timestamp"))))
                fz1 = fp["1:Fz"].values.astype(float)
                fz2 = fp["2:Fz"].values.astype(float)
                raw_edges = np.where(np.diff((fz1 > 20).astype(int)) == 1)[0] + 1
                rec["v1_edges_plate1"] = int((raw_edges > 10 * FS_FP).sum())
                # v3: remove each plate's offset (median outside contacts)
                fy1 = fp["1:Fy"].values.astype(float)
                c01, c02 = fp_contacts(fz1, warmup_s=0), fp_contacts(fz2, warmup_s=0)
                fz1, rec["base_fz1"] = baseline_correct(fz1, c01)
                fz2, rec["base_fz2"] = baseline_correct(fz2, c02)
                fy1, rec["base_fy1"] = baseline_correct(fy1, c01)
                c1, c2 = fp_contacts(fz1), fp_contacts(fz2)
                rec["fp_contacts_p1"], rec["fp_contacts_p2"] = len(c1), len(c2)
                rec["fp_stance_med_s"] = np.median([(b - a) / FS_FP for a, b, _ in c1]) if c1 else np.nan
                hs_fp[1] = np.array([a for a, _, _ in c1])
                hs_fp[2] = np.array([a for a, _, _ in c2])
                st1 = clean_strides(hs_fp[1], FS_FP)
                rec["fp_strides_p1"] = len(st1)
                rec["fp_stride_med_s"] = np.median([(b - a) / FS_FP for a, b in st1]) if st1 else np.nan
                pk1 = [p for _, _, p in c1]
                pk2 = [p for _, _, p in c2]
                if len(pk1) >= 5 and len(pk2) >= 5:
                    xl, xr = np.mean(pk1), np.mean(pk2)
                    fai[(sid, sp)] = (max(xl, xr) - min(xl, xr)) / (0.5 * (xl + xr)) * 100
                if len(st1) >= 5:
                    bw = mass * G
                    chans = {"Fz": fz1 / bw, "Fy": fy1 / bw,
                             "COPx": fp["1:COPx"].values.astype(float), "Fz_N": fz1}
                    fpw[(sid, sp)] = {k: np.mean([tn(v[a:b]) for a, b in st1], axis=0) for k, v in chans.items()}
            # ---------------- IMU ----------------
            if pos is not None and ang is not None:
                rec["imu_s"] = len(pos) / FS_IMU
                hsL, perL, sfL = imu_heel_strikes(pos, "Left")
                hsR, perR, sfR = imu_heel_strikes(pos, "Right")
                rec["imu_stance_frac_L"] = sfL
                stL, stR = clean_strides(hsL, FS_IMU), clean_strides(hsR, FS_IMU)
                rec["imu_strides_L"], rec["imu_strides_R"] = len(stL), len(stR)
                rec["imu_stride_med_s"] = np.median([(b - a) / FS_IMU for a, b in stL]) if stL else np.nan
                if len(stL) >= 5 and len(stR) >= 5:
                    d = {}
                    for j, (col, _) in JOINTS.items():
                        r = ang[f"Right {col}"].values
                        l = ang[f"Left {col}"].values
                        R = np.mean([tn(r[a:b]) for a, b in stR], axis=0)
                        L = np.mean([tn(l[a:b]) for a, b in stL], axis=0)
                        # v1 definition: both legs inside the same LEFT-HS window
                        v1 = np.mean([tn(r[a:b]) - tn(l[a:b]) for a, b in stL], axis=0)
                        d[j] = dict(R=R, L=L, KA=R - L, KA_v1=v1)
                    kin[(sid, sp)] = d
                # ---- alignment diagnostic: IMU left HS vs FP plate-1 HS ----
                if fp is not None and len(hsL) > 5 and len(hs_fp.get(1, [])) > 5:
                    t_imu = hsL / FS_IMU
                    t_fp = hs_fp[1] / FS_FP
                    def mad_at(lag):
                        tt = t_imu + lag
                        idx = np.clip(np.searchsorted(t_fp, tt), 1, len(t_fp) - 1)
                        dd = np.minimum(np.abs(t_fp[idx] - tt), np.abs(t_fp[idx - 1] - tt))
                        return np.median(dd)
                    lags = np.arange(-3, 3.0001, 1 / FS_IMU)
                    m = np.array([mad_at(x) for x in lags])
                    rec["sync_mad_lag0_ms"] = mad_at(0.0) * 1000
                    rec["sync_best_lag_s"] = float(lags[np.argmin(m)])
                    rec["sync_mad_best_ms"] = float(m.min()) * 1000
                    stride = rec.get("imu_stride_med_s", np.nan)
                    rec["sync_mad_lag0_pct_stride"] = mad_at(0.0) / stride * 100 if stride == stride else np.nan
            # ---- EMG vs FP alignment diagnostic ----
            emg = load_emg(sid, sp)
            if emg is not None and fp is not None and len(hs_fp.get(1, [])) > 5:
                te = emg.iloc[:, 0].values
                col = [c for c in emg.columns if "L GASTROCNEMIUS MEDIAL" in c][0]
                g = emg[col].values.astype(float)
                stance = np.zeros(len(fp))
                for a, b, _ in fp_contacts(fp["1:Fz"].values.astype(float)):
                    stance[a:b] = 1
                s_at = lambda lag: np.interp(te + lag, np.arange(len(fp)) / FS_FP, stance)
                lags = np.arange(-1.5, 1.5001, 0.0625)
                cc = np.array([np.corrcoef(s_at(x), g)[0, 1] for x in lags])
                emg_rows.append(dict(sid=sid, speed=sp, emg_s=te[-1], cc_lag0=float(np.corrcoef(s_at(0), g)[0, 1]),
                                     best_lag_s=float(lags[np.argmax(cc)]), cc_best=float(cc.max())))
            trials.append(rec)
    import pickle
    with open(os.path.join(OUT, "waveforms_v3.pkl"), "wb") as fh:
        pickle.dump(dict(kin=kin, fpw=fpw, fai=fai), fh)
    tr = pd.DataFrame(trials)
    tr.to_csv(os.path.join(OUT, "trial_diagnostics.csv"), index=False)
    em = pd.DataFrame(emg_rows)
    em.to_csv(os.path.join(OUT, "emg_fp_alignment.csv"), index=False)

    summary = {}
    # ---------------- diagnostics summary ----------------
    summary["fp_duration_s_range"] = [float(tr.fp_s.min()), float(tr.fp_s.max())]
    summary["imu_duration_s_range"] = [float(tr.imu_s.min()), float(tr.imu_s.max())]
    summary["fp_sync_channels_all_zero"] = bool((tr.fp_sync_nonzero.fillna(0) == 0).all())
    tr["v1_edges_per_debounced_contact"] = tr.v1_edges_plate1 / tr.fp_contacts_p1
    summary["v1_edges_per_true_contact_by_speed"] = tr.groupby("speed")["v1_edges_per_debounced_contact"].median().round(2).to_dict()
    summary["fp_vs_imu_stride_s_by_speed"] = tr.groupby("speed")[["fp_stride_med_s", "imu_stride_med_s"]].median().round(3).to_dict()
    summary["imu_stance_frac_median"] = float(tr.imu_stance_frac_L.median())
    s = tr.dropna(subset=["sync_mad_lag0_ms"])
    summary["imu_fp_sync"] = dict(
        n_trials=int(len(s)),
        median_abs_event_diff_lag0_ms=float(s.sync_mad_lag0_ms.median()),
        median_abs_event_diff_lag0_pct_stride=float(s.sync_mad_lag0_pct_stride.median()),
        best_lag_s_median=float(s.sync_best_lag_s.median()),
        best_lag_s_IQR=[float(s.sync_best_lag_s.quantile(.25)), float(s.sync_best_lag_s.quantile(.75))],
        best_lag_s_range=[float(s.sync_best_lag_s.min()), float(s.sync_best_lag_s.max())],
        frac_trials_lag0_within_50ms=float((s.sync_mad_lag0_ms < 50).mean()),
        median_abs_event_diff_bestlag_ms=float(s.sync_mad_best_ms.median()))
    summary["emg_fp_sync"] = dict(
        n_trials=int(len(em)),
        best_lag_s_median=float(em.best_lag_s.median()),
        best_lag_s_IQR=[float(em.best_lag_s.quantile(.25)), float(em.best_lag_s.quantile(.75))],
        frac_best_lag_within_0p2s=float((em.best_lag_s.abs() <= 0.2).mean()),
        cc_lag0_median=float(em.cc_lag0.median()), cc_best_median=float(em.cc_best.median()))

    # ---------------- KA analyses ----------------
    complete = [s_ for s_ in SUBJECTS if all((s_, sp) in kin for sp in SPEEDS)]
    summary["ka_n_complete"] = len(complete)
    summary["ka_excluded"] = [s_ for s_ in SUBJECTS if s_ not in complete]
    an = meta.loc[complete]
    summary["analysed_demographics"] = dict(
        male=int((an.sex == "M").sum()), female=int((an.sex == "F").sum()),
        age_mean=float(an.Age.mean()), age_sd=float(an.Age.std()), age_min=float(an.Age.min()), age_max=float(an.Age.max()),
        bmi_mean=float(an.BMI.mean()), bmi_sd=float(an.BMI.std()),
        bmi_groups=an.bmi_group.value_counts().reindex(["Normal", "Overweight", "Obese"]).astype(int).to_dict())
    summary["full_demographics"] = dict(
        male=int((meta.sex == "M").sum()), female=int((meta.sex == "F").sum()),
        age_mean=float(meta.Age.mean()), age_sd=float(meta.Age.std()), age_min=float(meta.Age.min()), age_max=float(meta.Age.max()),
        bmi_mean=float(meta.BMI.mean()), bmi_sd=float(meta.BMI.std()), bmi_min=float(meta.BMI.min()), bmi_max=float(meta.BMI.max()),
        bmi_groups=meta.bmi_group.value_counts().reindex(["Normal", "Overweight", "Obese"]).astype(int).to_dict())

    spm_rows, icc_rows, mag_rows = [], [], []
    n = len(complete)
    for j, (_, label) in JOINTS.items():
        arrs = [np.array([kin[(s_, sp)][j]["KA"] for s_ in complete]) for sp in SPEEDS]
        Y = np.vstack(arrs)
        A = np.repeat(np.arange(3), n)
        S = np.tile(np.arange(n), 3)
        Fi = spm1d.stats.anova1rm(Y, A, S).inference(0.05)
        v1 = [np.array([kin[(s_, sp)][j]["KA_v1"] for s_ in complete]) for sp in SPEEDS]
        Fi1 = spm1d.stats.anova1rm(np.vstack(v1), A, S).inference(0.05)
        spm_rows.append(dict(joint=label, n=n, v2_h0reject=bool(Fi.h0reject), v2_clusters=clusters_of(Fi),
                             v2_Fmax=float(Fi.z.max()), v1method_h0reject=bool(Fi1.h0reject),
                             v1method_clusters=clusters_of(Fi1), v1method_Fmax=float(Fi1.z.max())))
        # one-sample t vs 0 per speed (is there systematic group asymmetry?)
        for sp, arr in zip(SPEEDS, arrs):
            ti = spm1d.stats.ttest(arr, 0).inference(0.05, two_tailed=True)
            mag_rows.append(dict(joint=label, speed=sp, mean_abs_KA_v2=float(np.mean(np.abs(arr))),
                                 mean_abs_KA_v1method=float(np.mean(np.abs(v1[SPEEDS.index(sp)]))),
                                 KA_vs0_clusters=clusters_of(ti)))
        for metric, fn in [("mean|KA|", lambda k: np.mean(np.abs(k))),
                           ("static offset |mean KA|", lambda k: abs(np.mean(k))),
                           ("dynamic RMS (KA - mean)", lambda k: np.sqrt(np.mean((k - k.mean()) ** 2)))]:
            M = np.array([[fn(kin[(s_, sp)][j]["KA"]) for sp in SPEEDS] for s_ in complete])
            r = icc_all(M)
            r.update(joint=label, metric=metric, mean_by_speed=np.round(M.mean(0), 2).tolist())
            icc_rows.append(r)
        if True:  # figure
            fig, ax = plt.subplots(figsize=(6, 3.2))
            x = np.linspace(0, 100, 101)
            for sp, arr, c in zip(SPEEDS, arrs, ["#1f77b4", "#ff7f0e", "#d62728"]):
                m_, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(n)
                ax.plot(x, m_, color=c, label=f"{sp} m/s")
                ax.fill_between(x, m_ - se, m_ + se, color=c, alpha=.15)
            for c_ in Fi.clusters:
                ax.axvspan(float(c_.endpoints[0]), float(c_.endpoints[1]), ymin=.94, ymax=1, color="k")
            ax.axhline(0, color="grey", lw=.7, ls="--")
            ax.set_xlabel("Gait cycle (%)"); ax.set_ylabel("KA (deg)"); ax.set_title(f"{label} (n={n})")
            ax.legend(fontsize=7, frameon=False)
            fig.tight_layout(); fig.savefig(os.path.join(FIG, f"ka_{j}.png"), dpi=200); plt.close(fig)
    pd.DataFrame(spm_rows).to_csv(os.path.join(OUT, "ka_spm_rmanova.csv"), index=False)
    pd.DataFrame(icc_rows).to_csv(os.path.join(OUT, "ka_icc.csv"), index=False)
    pd.DataFrame(mag_rows).to_csv(os.path.join(OUT, "ka_magnitude_by_speed.csv"), index=False)

    # ---------------- KA vs FAI (subject-level, per speed) ----------------
    cor_rows = []
    for sp in SPEEDS:
        subs = [s_ for s_ in SUBJECTS if (s_, sp) in kin and (s_, sp) in fai]
        f = np.array([fai[(s_, sp)] for s_ in subs])
        for j, (_, label) in JOINTS.items():
            k = np.array([np.mean(np.abs(kin[(s_, sp)][j]["KA"])) for s_ in subs])
            r, p = stats.pearsonr(k, f)
            rs, ps = stats.spearmanr(k, f)
            cor_rows.append(dict(speed=sp, joint=label, n=len(subs), pearson_r=r, p=p, spearman_rho=rs, p_spearman=ps))
    cr = pd.DataFrame(cor_rows)
    cr["p_bonf12"] = np.minimum(cr.p * len(cr), 1)
    cr.to_csv(os.path.join(OUT, "ka_fai_correlation.csv"), index=False)
    fa = pd.DataFrame([dict(sid=k[0], speed=k[1], FAI=v) for k, v in fai.items()])
    summary["FAI_peakFz_by_speed_mean_sd"] = fa.groupby("speed").FAI.agg(["mean", "std", "count"]).round(2).to_dict()

    # ---------------- BMI (one-way SPM ANOVA per speed) ----------------
    bmi_rows = []
    for sp in SPEEDS:
        for ch in ["Fz", "Fy", "COPx", "Fz_N"]:
            groups = {g: [fpw[(s_, sp)][ch] for s_ in SUBJECTS if (s_, sp) in fpw and meta.loc[s_, "bmi_group"] == g]
                      for g in ["Normal", "Overweight", "Obese"]}
            Y = np.vstack([np.array(v) for v in groups.values()])
            A = np.concatenate([[i] * len(v) for i, v in enumerate(groups.values())])
            Fi = spm1d.stats.anova1(Y, A).inference(0.05)
            bmi_rows.append(dict(speed=sp, channel=ch + ("" if ch in ("COPx", "Fz_N") else " (BW)"),
                                 n=[len(v) for v in groups.values()], h0reject=bool(Fi.h0reject), clusters=clusters_of(Fi)))
        for j, (_, label) in JOINTS.items():
            if j == "hip_abd":
                continue
            groups = {g: [kin[(s_, sp)][j]["L"] for s_ in SUBJECTS if (s_, sp) in kin and meta.loc[s_, "bmi_group"] == g]
                      for g in ["Normal", "Overweight", "Obese"]}
            Y = np.vstack([np.array(v) for v in groups.values()])
            A = np.concatenate([[i] * len(v) for i, v in enumerate(groups.values())])
            Fi = spm1d.stats.anova1(Y, A).inference(0.05)
            bmi_rows.append(dict(speed=sp, channel=f"Left {label}", n=[len(v) for v in groups.values()],
                                 h0reject=bool(Fi.h0reject), clusters=clusters_of(Fi)))
    pd.DataFrame(bmi_rows).to_csv(os.path.join(OUT, "bmi_spm.csv"), index=False)

    # ---------------- Sex (two-sample SPM t per speed), FP only ----------------
    sex_rows = []
    for sp in SPEEDS:
        for ch in ["Fz", "Fy", "COPx", "Fz_N"]:
            M_ = np.array([fpw[(s_, sp)][ch] for s_ in SUBJECTS if (s_, sp) in fpw and meta.loc[s_, "sex"] == "M"])
            F_ = np.array([fpw[(s_, sp)][ch] for s_ in SUBJECTS if (s_, sp) in fpw and meta.loc[s_, "sex"] == "F"])
            ti = spm1d.stats.ttest2(M_, F_).inference(0.05, two_tailed=True)
            sex_rows.append(dict(speed=sp, channel=ch + ("" if ch in ("COPx", "Fz_N") else " (BW)"),
                                 n_m=len(M_), n_f=len(F_), h0reject=bool(ti.h0reject), clusters=clusters_of(ti)))
    pd.DataFrame(sex_rows).to_csv(os.path.join(OUT, "sex_spm_fp.csv"), index=False)

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(json.dumps(summary, indent=2, default=str))
    print(pd.DataFrame(spm_rows).to_string())
    print(pd.DataFrame(icc_rows)[["joint", "metric", "ICC_C1", "C1_lo", "C1_hi", "ICC_A1", "A1_lo", "A1_hi", "mean_by_speed"]].round(3).to_string())
    print(pd.DataFrame(mag_rows).round(2).to_string())
    print(cr.round(3).to_string())
    print(pd.DataFrame(bmi_rows).to_string())
    print(pd.DataFrame(sex_rows).to_string())


if __name__ == "__main__":
    main()
