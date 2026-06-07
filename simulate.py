"""
CS NMOS Amplifier DC Simulation — main script.

Circuit parameters:
  VDD=5V, RD=10kOhm
  NMOS M0: Vth=0.8V, Kp=100uA/V^2, lambda=0.02V^-1, W/L=10
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from spice_sim.models.nmos import NMOSModel
from spice_sim.analysis.dc_sweep import DCSweep

# ── Circuit parameters ─────────────────────────────────────────────────────
VDD   = 5.0       # Supply voltage (V)
R_D   = 10e3      # Drain load resistance (Ohm)
VTH   = 0.8       # Threshold voltage (V)
KP    = 100e-6    # Process transconductance (A/V^2)
LAM   = 0.02      # Channel-length modulation coefficient (V^-1)
WL    = 10.0      # W/L ratio
DELTA = 0.02      # Smooth-transition parameter (V)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device model & sweep engine ────────────────────────────────────────────
nmos  = NMOSModel(vth=VTH, kp=KP, lam=LAM, wl=WL, delta=DELTA)
sweep = DCSweep(nmos, R_D, VDD)

# ── Sweep arrays ───────────────────────────────────────────────────────────
# Coarse: 0-5V step 0.25V (required by assignment)
vin_coarse = np.arange(0, 5.001, 0.25)

# Fine near Vth (cutoff->active), step 5mV
vin_fine_off = np.arange(0.5, 1.2, 0.005)

# Fine near linear/saturation boundary, step 2mV
vin_fine_trans = np.arange(1.1, 2.5, 0.002)

# Combined full sweep for continuous curves
vin_full = np.sort(np.unique(np.concatenate([
    np.arange(0, 5.001, 0.01),
    vin_fine_off,
    vin_fine_trans,
])))

# ── Run simulations ────────────────────────────────────────────────────────
print("=" * 60)
print("DC Sweep Simulation (coarse, 0-5V, step 0.25V)")
print("=" * 60)
res_coarse = sweep.run(vin_coarse, verbose=True)
data_coarse = DCSweep.to_arrays(res_coarse)

print("\nFine-step sweep (near Vth and linear/saturation transition)...")
res_full = sweep.run(vin_full, verbose=False)
data_full = DCSweep.to_arrays(res_full)

# ── Locate transition point (VDS = Vdsat = VGS - Vth) ─────────────────────
idx_trans  = np.argmin(np.abs(data_full["vout"] - (data_full["vin"] - VTH)))
vin_trans_pt  = data_full["vin"][idx_trans]
vout_trans_pt = data_full["vout"][idx_trans]
print(f"\nLinear/Saturation boundary: VIN ~ {vin_trans_pt:.3f}V, VOUT ~ {vout_trans_pt:.3f}V")
print(f"Cutoff->Active boundary:    VIN ~ {VTH:.2f}V")

# ── Figure 1: VOUT vs VIN (main plot) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(data_full["vin"], data_full["vout"],
        color="#1f77b4", linewidth=2, label="Simulator (continuous)")
ax.scatter(data_coarse["vin"], data_coarse["vout"],
           color="red", s=40, zorder=5, label="Coarse samples (0.25V step)")
ax.axvline(VTH, color="gray", linestyle="--", linewidth=1, alpha=0.7,
           label=f"Vth = {VTH}V")
ax.axvline(vin_trans_pt, color="orange", linestyle="--", linewidth=1, alpha=0.8,
           label=f"Lin/Sat boundary ~ {vin_trans_pt:.2f}V")
ax.annotate(f"VOUT={vout_trans_pt:.3f}V",
            xy=(vin_trans_pt, vout_trans_pt),
            xytext=(vin_trans_pt + 0.5, vout_trans_pt + 0.5),
            arrowprops=dict(arrowstyle="->", color="orange"),
            fontsize=9, color="orange")
ax.set_xlabel("VIN (V)", fontsize=12)
ax.set_ylabel("VOUT (V)", fontsize=12)
ax.set_title("CS NMOS Amplifier: VOUT vs VIN (DC Sweep)", fontsize=13)
ax.set_xlim(0, 5)
ax.set_ylim(-0.1, VDD + 0.3)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "vout_vs_vin.png"), dpi=150)
print("\nSaved: results/vout_vs_vin.png")
plt.close()

# ── Figure 2: Zoom near critical transition points ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: near Vth
mask_off = (data_full["vin"] >= 0.5) & (data_full["vin"] <= 1.2)
ax = axes[0]
ax.plot(data_full["vin"][mask_off], data_full["vout"][mask_off],
        color="#1f77b4", linewidth=2)
ax.axvline(VTH, color="gray", linestyle="--", linewidth=1.5, label=f"Vth={VTH}V")
ax.set_xlabel("VIN (V)", fontsize=11)
ax.set_ylabel("VOUT (V)", fontsize=11)
ax.set_title("Zoom: Cutoff -> Active (near Vth)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: near linear/saturation boundary
margin = 0.5
mask_tr = ((data_full["vin"] >= vin_trans_pt - margin) &
           (data_full["vin"] <= vin_trans_pt + margin))
ax = axes[1]
ax.plot(data_full["vin"][mask_tr], data_full["vout"][mask_tr],
        color="#1f77b4", linewidth=2)
ax.axvline(vin_trans_pt, color="orange", linestyle="--", linewidth=1.5,
           label=f"Boundary ~ {vin_trans_pt:.2f}V")
ax.annotate(f"({vin_trans_pt:.2f}V, {vout_trans_pt:.3f}V)",
            xy=(vin_trans_pt, vout_trans_pt),
            xytext=(vin_trans_pt + 0.1, vout_trans_pt + 0.1),
            arrowprops=dict(arrowstyle="->", color="orange"),
            fontsize=9, color="orange")
ax.set_xlabel("VIN (V)", fontsize=11)
ax.set_ylabel("VOUT (V)", fontsize=11)
ax.set_title("Zoom: Linear / Saturation Transition", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle("Fine-Step Sweep Near Critical Points (2~5 mV step)", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "vout_vs_vin_zoom.png"),
            dpi=150, bbox_inches="tight")
print("Saved: results/vout_vs_vin_zoom.png")
plt.close()

# ── Figure 3: ID vs VIN ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data_full["vin"], data_full["id"] * 1e3,
        color="#2ca02c", linewidth=2, label="ID continuous")
ax.scatter(data_coarse["vin"], data_coarse["id"] * 1e3,
           color="red", s=40, zorder=5, label="Coarse samples")
ax.axvline(VTH, color="gray", linestyle="--", linewidth=1, alpha=0.7,
           label=f"Vth={VTH}V")
ax.set_xlabel("VIN (V)", fontsize=12)
ax.set_ylabel("ID (mA)", fontsize=12)
ax.set_title("Drain Current ID vs VIN", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "id_vs_vin.png"), dpi=150)
print("Saved: results/id_vs_vin.png")
plt.close()

# ── Figure 4: Small-signal parameters gm / ro / |Av| vs VIN ───────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax = axes[0]
ax.plot(data_full["vin"], data_full["gm"] * 1e3, color="#d62728", linewidth=2)
ax.set_ylabel("gm (mA/V)", fontsize=11)
ax.set_title("Small-Signal Parameters vs VIN", fontsize=13)
ax.grid(True, alpha=0.3)

ax = axes[1]
ro_kOhm = np.minimum(data_full["ro"] / 1e3, 1e4)
ax.plot(data_full["vin"], ro_kOhm, color="#9467bd", linewidth=2)
ax.set_ylabel("ro (kOhm, clipped at 10 GOhm)", fontsize=11)
ax.set_ylim(0, min(ro_kOhm[ro_kOhm < 1e3].max() * 1.5 if any(ro_kOhm < 1e3) else 500, 2000))
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(data_full["vin"], np.abs(data_full["av"]), color="#8c564b", linewidth=2)
ax.set_xlabel("VIN (V)", fontsize=12)
ax.set_ylabel("|Av| = gm*(ro||RD)", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "smallsig_vs_vin.png"), dpi=150)
print("Saved: results/smallsig_vs_vin.png")
plt.close()

# ── Figure 5: Newton-Raphson iteration count ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(data_full["vin"], data_full["n_iter"], width=0.008,
       color="#17becf", alpha=0.8)
ax.set_xlabel("VIN (V)", fontsize=12)
ax.set_ylabel("Newton Iterations", fontsize=11)
ax.set_title("Newton-Raphson Iterations per Operating Point", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "newton_iters.png"), dpi=150)
print("Saved: results/newton_iters.png")
plt.close()

# ── Terminal summary table ─────────────────────────────────────────────────
print("\n" + "=" * 75)
print("Coarse Sweep Summary (0.25V step)")
print(f"{'VIN(V)':>8} {'VOUT(V)':>10} {'ID(mA)':>10} "
      f"{'gm(mA/V)':>10} {'|Av|':>8} {'Region':>12} {'Iter':>5}")
print("-" * 75)
for r in res_coarse:
    print(f"{r['vin']:>8.2f} {r['vout']:>10.5f} {r['id']*1e3:>10.5f} "
          f"{r['gm']*1e3:>10.5f} {abs(r['av']):>8.3f} "
          f"{r['region']:>12s} {r['n_iter']:>5d}")
print("=" * 75)
print(f"\nAll figures saved to: {RESULTS_DIR}")

# (overview figure generated after Level-2 data is ready — see end of script)

# ═══════════════════════════════════════════════════════════════════════════
# Level-2 模型对比
# 逐一开启高阶效应，观察每种效应对 VOUT-VIN 曲线的影响
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- Level-2 enhanced model sweep ---")

# Level-2 参数（典型值）
L2_THETA  = 0.10   # 迁移率衰减系数 (V^-1)
L2_ETA    = 0.03   # DIBL 系数 (V/V)
L2_GAMMA  = 0.50   # 体效应系数 (V^0.5)
L2_PHI_F  = 0.70   # 2φF (V)
L2_N_SUB  = 1.30   # 亚阈值斜率因子
L2_RS     = 50.0   # 源极寄生电阻 (Ω)
L2_RD     = 50.0   # 漏极寄生电阻 (Ω)

# 各效应单独开启 + 全开，与 Level-1 对比
configs = {
    "Level-1 (baseline)":  dict(theta=0,        eta=0,       gamma=0,      n_sub=None,   Rs=0,     Rd=0),
    "2A: Subthreshold":    dict(theta=0,        eta=0,       gamma=0,      n_sub=L2_N_SUB, Rs=0,   Rd=0),
    "2B: Mob. degrad.":    dict(theta=L2_THETA, eta=0,       gamma=0,      n_sub=None,   Rs=0,     Rd=0),
    "2C: DIBL":            dict(theta=0,        eta=L2_ETA,  gamma=0,      n_sub=None,   Rs=0,     Rd=0),
    "2D: Body effect":     dict(theta=0,        eta=0,       gamma=L2_GAMMA, n_sub=None, Rs=0,     Rd=0),
    "2E: Rs/Rd=50Ω":       dict(theta=0,        eta=0,       gamma=0,      n_sub=None,   Rs=L2_RS, Rd=L2_RD),
    "All Level-2":         dict(theta=L2_THETA, eta=L2_ETA,  gamma=L2_GAMMA, n_sub=L2_N_SUB, Rs=L2_RS, Rd=L2_RD),
}
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
styles = ["-", "--", "--", "--", "--", "--", "-"]
lws    = [2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5]

all_data = {}
for label, cfg in configs.items():
    Rs_val = cfg.pop("Rs", 0.0)
    Rd_val = cfg.pop("Rd", 0.0)
    m_cfg = NMOSModel(vth=VTH, kp=KP, lam=LAM, wl=WL, delta=DELTA, **cfg)
    sw    = DCSweep(m_cfg, R_D, VDD, Rs=Rs_val, Rd=Rd_val)
    res   = sw.run(vin_full, verbose=False)
    all_data[label] = DCSweep.to_arrays(res)
    print(f"  {label}: done")

# ── Figure 7: Level-2 VOUT comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax.plot(data["vin"], data["vout"], color=c, linestyle=ls, linewidth=lw, label=label)
ax.axvline(VTH, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("VIN (V)"); ax.set_ylabel("VOUT (V)")
ax.set_title("VOUT vs VIN: Effect of Level-2 Enhancements")
ax.set_xlim(0, 5); ax.set_ylim(-0.1, VDD + 0.3)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# Zoom on saturation region (transition area) for clearer differentiation
ax = axes[1]
vin_zoom = (vin_full >= 0.5) & (vin_full <= 2.5)
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax.plot(data["vin"][vin_zoom], data["vout"][vin_zoom],
            color=c, linestyle=ls, linewidth=lw, label=label)
ax.axvline(VTH, color="gray", linestyle=":", linewidth=1, alpha=0.6,
           label=f"Vth={VTH}V")
ax.set_xlabel("VIN (V)"); ax.set_ylabel("VOUT (V)")
ax.set_title("Zoom: 0.5–2.5 V (Saturation Region)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f"Level-2 Model Effects  "
             f"(θ={L2_THETA}/V, η={L2_ETA}, γ={L2_GAMMA}V^0.5, "
             f"n={L2_N_SUB}, Rs=Rd={L2_RS}Ω)",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "level2_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/level2_comparison.png")

# ── Figure 8: Level-2 gm comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax.plot(data["vin"], data["gm"] * 1e3, color=c, linestyle=ls, linewidth=lw, label=label)
ax.set_xlabel("VIN (V)"); ax.set_ylabel("gm (mA/V)")
ax.set_title("Transconductance gm: Level-1 vs Level-2")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax.plot(data["vin"], data["id"] * 1e3, color=c, linestyle=ls, linewidth=lw, label=label)
ax.set_xlabel("VIN (V)"); ax.set_ylabel("ID (mA)")
ax.set_title("Drain Current ID: Level-1 vs Level-2")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "level2_gm_id.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/level2_gm_id.png")

# ── Figure 6 (updated): Full overview — Level-1 results + Level-2 comparison
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(18, 19))
fig.suptitle(
    "CS NMOS Amplifier — DC Simulation Full Overview\n"
    f"VDD={VDD}V  RD={R_D/1e3:.0f}kΩ  Vth={VTH}V  "
    f"Kp={KP*1e6:.0f}μA/V²  λ={LAM}V⁻¹  W/L={WL:.0f}",
    fontsize=14, y=0.995)

gs = GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.35)

# ── Row 0: VOUT vs VIN (wide) + ID vs VIN ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(data_full["vin"], data_full["vout"],
         color="#1f77b4", linewidth=2, label="Simulator")
ax1.scatter(data_coarse["vin"], data_coarse["vout"],
            color="red", s=30, zorder=5, label="0.25V step")
ax1.axvline(VTH, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax1.axvline(vin_trans_pt, color="orange", linestyle="--", linewidth=1, alpha=0.8)
ax1.set_xlabel("VIN (V)"); ax1.set_ylabel("VOUT (V)")
ax1.set_title("VOUT vs VIN (Level-1)"); ax1.set_xlim(0, 5); ax1.set_ylim(-0.1, VDD + 0.3)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(data_full["vin"], data_full["id"] * 1e3, color="#2ca02c", linewidth=2)
ax2.scatter(data_coarse["vin"], data_coarse["id"] * 1e3, color="red", s=20, zorder=5)
ax2.axvline(VTH, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.set_xlabel("VIN (V)"); ax2.set_ylabel("ID (mA)")
ax2.set_title("ID vs VIN"); ax2.grid(True, alpha=0.3)

# ── Row 1: Zoom cutoff / lin-sat / Newton iters ───────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
mask_off = (data_full["vin"] >= 0.5) & (data_full["vin"] <= 1.2)
ax3.plot(data_full["vin"][mask_off], data_full["vout"][mask_off],
         color="#1f77b4", linewidth=2)
ax3.axvline(VTH, color="gray", linestyle="--", linewidth=1.5)
ax3.set_xlabel("VIN (V)"); ax3.set_ylabel("VOUT (V)")
ax3.set_title(f"Zoom: Cutoff->Active (Vth={VTH}V)"); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
margin = 0.5
mask_tr = ((data_full["vin"] >= vin_trans_pt - margin) &
           (data_full["vin"] <= vin_trans_pt + margin))
ax4.plot(data_full["vin"][mask_tr], data_full["vout"][mask_tr],
         color="#1f77b4", linewidth=2)
ax4.axvline(vin_trans_pt, color="orange", linestyle="--", linewidth=1.5)
ax4.annotate(f"({vin_trans_pt:.2f}V,\n{vout_trans_pt:.3f}V)",
             xy=(vin_trans_pt, vout_trans_pt),
             xytext=(vin_trans_pt + 0.12, vout_trans_pt + 0.08),
             arrowprops=dict(arrowstyle="->", color="orange"), fontsize=8, color="orange")
ax4.set_xlabel("VIN (V)"); ax4.set_ylabel("VOUT (V)")
ax4.set_title("Zoom: Lin/Sat Boundary"); ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
ax5.bar(data_full["vin"], data_full["n_iter"], width=0.008, color="#17becf", alpha=0.8)
ax5.set_xlabel("VIN (V)"); ax5.set_ylabel("Iterations")
ax5.set_title("Newton-Raphson Iterations"); ax5.grid(True, alpha=0.3, axis="y")

# ── Row 2: Small-signal parameters ───────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(data_full["vin"], data_full["gm"] * 1e3, color="#d62728", linewidth=2)
ax6.set_xlabel("VIN (V)"); ax6.set_ylabel("gm (mA/V)")
ax6.set_title("Transconductance gm"); ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[2, 1])
ro_kOhm = np.minimum(data_full["ro"] / 1e3, 1e4)
valid = ro_kOhm < 1e3
ax7.plot(data_full["vin"], ro_kOhm, color="#9467bd", linewidth=2)
ax7.set_ylim(0, ro_kOhm[valid].max() * 1.5 if valid.any() else 500)
ax7.set_xlabel("VIN (V)"); ax7.set_ylabel("ro (kΩ)")
ax7.set_title("Output Resistance ro"); ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 2])
ax8.plot(data_full["vin"], np.abs(data_full["av"]), color="#8c564b", linewidth=2)
ax8.set_xlabel("VIN (V)"); ax8.set_ylabel("|Av|")
ax8.set_title("Voltage Gain |Av| = gm·(ro||RD)"); ax8.grid(True, alpha=0.3)

# ── Row 3: Level-2 comparison (wide VOUT + ID) ───────────────────────────
ax9 = fig.add_subplot(gs[3, :2])
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax9.plot(data["vin"], data["vout"], color=c, linestyle=ls, linewidth=lw, label=label)
ax9.axvline(VTH, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax9.set_xlabel("VIN (V)"); ax9.set_ylabel("VOUT (V)")
ax9.set_title("Level-2 Model Effects: VOUT vs VIN")
ax9.set_xlim(0, 5); ax9.set_ylim(-0.1, VDD + 0.3)
ax9.legend(fontsize=7, ncol=2, loc="upper right"); ax9.grid(True, alpha=0.3)

ax10 = fig.add_subplot(gs[3, 2])
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax10.plot(data["vin"], data["id"] * 1e3, color=c, linestyle=ls, linewidth=lw, label=label)
ax10.set_xlabel("VIN (V)"); ax10.set_ylabel("ID (mA)")
ax10.set_title("Level-2 Effects: ID vs VIN")
ax10.legend(fontsize=6); ax10.grid(True, alpha=0.3)

fig.savefig(os.path.join(RESULTS_DIR, "overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/overview.png  (updated with Level-2 row)")
