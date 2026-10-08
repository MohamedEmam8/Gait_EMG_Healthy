"""Manuscript figures for v3 (reads results_v3/waveforms_v3.pkl written by biomech_v3.py)."""
import os
import pickle
import numpy as np
import pandas as pd
import spm1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from biomech_v3 import OUT, SPEEDS, SUBJECTS, JOINTS, load_meta

FIGDIR = os.path.join(os.path.dirname(OUT), "figures_v3")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {0.5: "#1b9e77", 0.75: "#7570b3", 1.0: "#d95f02"}
X = np.linspace(0, 100, 101)

d = pickle.load(open(os.path.join(OUT, "waveforms_v3.pkl"), "rb"))
kin, fpw = d["kin"], d["fpw"]
meta = load_meta()
complete = [s for s in SUBJECTS if all((s, sp) in kin for sp in SPEEDS)]
n = len(complete)

# ── Figure 1: KA(t) by speed, 4 joints, SPM RM-ANOVA bars ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
for ax, (j, (_, label)) in zip(axes.ravel(), JOINTS.items()):
    arrs = [np.array([kin[(s, sp)][j]["KA"] for s in complete]) for sp in SPEEDS]
    Fi = spm1d.stats.anova1rm(np.vstack(arrs), np.repeat(np.arange(3), n), np.tile(np.arange(n), 3)).inference(0.05)
    for sp, a in zip(SPEEDS, arrs):
        m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(n)
        ax.plot(X, m, color=COL[sp], lw=1.2, label=f"{sp} m/s")
        ax.fill_between(X, m - se, m + se, color=COL[sp], alpha=.18, lw=0)
    for c in Fi.clusters:
        ax.axvspan(float(c.endpoints[0]), float(c.endpoints[1]), ymin=.95, ymax=1, color="k")
    ax.axhline(0, color="grey", lw=.6, ls="--")
    ax.set_title(label, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylabel("KA (deg)")
    ax.set_xlabel("Gait cycle (%)")
axes[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig1_ka_waveforms.png"), dpi=300)
plt.close(fig)

# ── Figure 2: ICC(C,1) and ICC(A,1) with 95 % CI for three KA summaries ─────
icc = pd.read_csv(os.path.join(OUT, "ka_icc.csv"))
metrics = ["mean|KA|", "static offset |mean KA|", "dynamic RMS (KA - mean)"]
labels = [v[1] for v in JOINTS.values()]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, met in zip(axes, metrics):
    sub = icc[icc.metric == met].set_index("joint").loc[labels]
    y = np.arange(len(labels))
    ax.errorbar(sub.ICC_C1, y + .12, xerr=[sub.ICC_C1 - sub.C1_lo, sub.C1_hi - sub.ICC_C1], fmt="o", ms=3.5,
                color="k", capsize=2, label="ICC(C,1)")
    ax.errorbar(sub.ICC_A1, y - .12, xerr=[sub.ICC_A1 - sub.A1_lo, sub.A1_hi - sub.ICC_A1], fmt="s", ms=3.5,
                color="grey", capsize=2, label="ICC(A,1)")
    for v in (0.5, 0.75, 0.9):
        ax.axvline(v, color="lightgrey", lw=.6, ls=":")
    ax.set_xlim(0.3, 1.0)
    ax.set_title(met, fontsize=8)
    ax.set_xlabel("ICC (95% CI)")
axes[0].set_yticks(np.arange(len(labels)))
axes[0].set_yticklabels(labels)
axes[0].invert_yaxis()
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig2_icc.png"), dpi=300)
plt.close(fig)

# ── Figure 3: vertical GRF in N vs body-weight units by BMI group (1.0 m/s),
#              and left ankle angle by BMI group (1.0 m/s) ─────────────────
gcol = {"Normal": "#4daf4a", "Overweight": "#ff7f00", "Obese": "#e41a1c"}
sp = 1.0
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, ch, yl in [(axes[0], "Fz_N", "Fz (N)"), (axes[1], "Fz", "Fz (body weights)")]:
    fgroups = []
    for g in gcol:
        a = np.array([fpw[(s, sp)][ch] for s in SUBJECTS if (s, sp) in fpw and meta.loc[s, "bmi_group"] == g])
        fgroups.append(a)
        ax.plot(X, a.mean(0), color=gcol[g], lw=1.2, label=f"{g} (n={len(a)})")
        ax.fill_between(X, a.mean(0) - a.std(0, ddof=1) / np.sqrt(len(a)), a.mean(0) + a.std(0, ddof=1) / np.sqrt(len(a)),
                        color=gcol[g], alpha=.18, lw=0)
    if ch == "Fz":   # v3: mark BW-normalised clusters; skip 0-2 % / 98-100 % segmentation-edge points
        Fz_i = spm1d.stats.anova1(np.vstack(fgroups), np.concatenate([[i] * len(a) for i, a in enumerate(fgroups)])).inference(0.05)
        for c in Fz_i.clusters:
            e0, e1 = float(c.endpoints[0]), float(c.endpoints[1])
            if e1 > 2 and e0 < 98:
                ax.axvspan(e0, e1, ymin=.95, ymax=1, color="k")
    ax.set_ylabel(yl); ax.set_xlabel("Gait cycle (%)"); ax.set_xlim(0, 100)
ax = axes[2]
groups = []
for g in gcol:
    a = np.array([kin[(s, sp)]["ankle"]["L"] for s in SUBJECTS if (s, sp) in kin and meta.loc[s, "bmi_group"] == g])
    groups.append(a)
    ax.plot(X, a.mean(0), color=gcol[g], lw=1.2)
    ax.fill_between(X, a.mean(0) - a.std(0, ddof=1) / np.sqrt(len(a)), a.mean(0) + a.std(0, ddof=1) / np.sqrt(len(a)),
                    color=gcol[g], alpha=.18, lw=0)
Fi = spm1d.stats.anova1(np.vstack(groups), np.concatenate([[i] * len(a) for i, a in enumerate(groups)])).inference(0.05)
for c in Fi.clusters:
    ax.axvspan(float(c.endpoints[0]), float(c.endpoints[1]), ymin=.95, ymax=1, color="k")
ax.set_ylabel("Left ankle DF(+)/PF(-) (deg)"); ax.set_xlabel("Gait cycle (%)"); ax.set_xlim(0, 100)
axes[0].legend(frameon=False, fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.45))
for a_, t in zip(axes, ["A", "B", "C"]):
    a_.text(-0.2, 1.05, t, transform=a_.transAxes, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig3_bmi.png"), dpi=300)
plt.close(fig)
print("figures written to", FIGDIR)
