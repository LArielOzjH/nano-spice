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

# (overview figure generated after secondary-effects data is ready — see end of script)

# ═══════════════════════════════════════════════════════════════════════════
# 二阶物理效应对比
# 逐一开启各效应，观察对 VOUT-VIN 曲线的影响
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- Secondary physical effects sweep ---")

# 二阶效应参数（典型值）
L2_THETA  = 0.10   # 迁移率衰减系数 θ (V^-1)
L2_ETA    = 0.03   # DIBL 系数 η (V/V)
L2_GAMMA  = 0.50   # 体效应系数 γ (V^0.5)
L2_PHI_F  = 0.70   # 2φF (V)
L2_N_SUB  = 1.30   # 亚阈值斜率因子 n
L2_RS     = 50.0   # 源极寄生电阻 (Ω)
L2_RD     = 50.0   # 漏极寄生电阻 (Ω)

# 各效应单独开启 + 全开，与基础模型对比
configs = {
    "Base model":                dict(theta=0,        eta=0,       gamma=0,        n_sub=None,     Rs=0,     Rd=0),
    "Subthreshold conduction":   dict(theta=0,        eta=0,       gamma=0,        n_sub=L2_N_SUB, Rs=0,     Rd=0),
    "Mobility degradation":      dict(theta=L2_THETA, eta=0,       gamma=0,        n_sub=None,     Rs=0,     Rd=0),
    "DIBL":                      dict(theta=0,        eta=L2_ETA,  gamma=0,        n_sub=None,     Rs=0,     Rd=0),
    "Body effect (γ)":           dict(theta=0,        eta=0,       gamma=L2_GAMMA, n_sub=None,     Rs=0,     Rd=0),
    "Parasitic Rs/Rd":           dict(theta=0,        eta=0,       gamma=0,        n_sub=None,     Rs=L2_RS, Rd=L2_RD),
    "All secondary effects":     dict(theta=L2_THETA, eta=L2_ETA,  gamma=L2_GAMMA, n_sub=L2_N_SUB, Rs=L2_RS, Rd=L2_RD),
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

# ── Figure 7: Secondary effects VOUT comparison ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax.plot(data["vin"], data["vout"], color=c, linestyle=ls, linewidth=lw, label=label)
ax.axvline(VTH, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("VIN (V)"); ax.set_ylabel("VOUT (V)")
ax.set_title("VOUT vs VIN: Secondary Physical Effects")
ax.set_xlim(0, 5); ax.set_ylim(-0.1, VDD + 0.3)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

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

plt.suptitle(f"Secondary Physical Effects  "
             f"(θ={L2_THETA}/V, η={L2_ETA}, γ={L2_GAMMA}V^0.5, "
             f"n={L2_N_SUB}, Rs=Rd={L2_RS}Ω)",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "secondary_effects_vout.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/secondary_effects_vout.png")

# ── Figure 8: Secondary effects gm/ID comparison ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax.plot(data["vin"], data["gm"] * 1e3, color=c, linestyle=ls, linewidth=lw, label=label)
ax.set_xlabel("VIN (V)"); ax.set_ylabel("gm (mA/V)")
ax.set_title("Transconductance gm: Base vs Secondary Effects")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax.plot(data["vin"], data["id"] * 1e3, color=c, linestyle=ls, linewidth=lw, label=label)
ax.set_xlabel("VIN (V)"); ax.set_ylabel("ID (mA)")
ax.set_title("Drain Current ID: Base vs Secondary Effects")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "secondary_effects_gm_id.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/secondary_effects_gm_id.png")

# ── Figure 6 (updated): Full overview — base model + secondary effects
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
ax1.set_title("VOUT vs VIN (base model)"); ax1.set_xlim(0, 5); ax1.set_ylim(-0.1, VDD + 0.3)
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

# ── Row 3: Secondary effects comparison (wide VOUT + ID) ─────────────────
ax9 = fig.add_subplot(gs[3, :2])
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax9.plot(data["vin"], data["vout"], color=c, linestyle=ls, linewidth=lw, label=label)
ax9.axvline(VTH, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax9.set_xlabel("VIN (V)"); ax9.set_ylabel("VOUT (V)")
ax9.set_title("Secondary Physical Effects: VOUT vs VIN")
ax9.set_xlim(0, 5); ax9.set_ylim(-0.1, VDD + 0.3)
ax9.legend(fontsize=7, ncol=2, loc="upper right"); ax9.grid(True, alpha=0.3)

ax10 = fig.add_subplot(gs[3, 2])
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    ax10.plot(data["vin"], data["id"] * 1e3, color=c, linestyle=ls, linewidth=lw, label=label)
ax10.set_xlabel("VIN (V)"); ax10.set_ylabel("ID (mA)")
ax10.set_title("Secondary Effects: ID vs VIN")
ax10.legend(fontsize=6); ax10.grid(True, alpha=0.3)

fig.savefig(os.path.join(RESULTS_DIR, "overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/overview.png  (with secondary effects row)")

# ═══════════════════════════════════════════════════════════════════════════
# AC 小信号频率分析
#
# 将 DC MNA 扩展到复数域：Y(jω) = G + jω·C
# 栅极电容模型（第4章，电荷划分）：
#   饱和区 : Cgs = 2/3·Cox·W·L + Cgso，Cgd = Cgdo
#   线性区 : Cgs = 1/2·Cox·W·L + Cgso，Cgd = 1/2·Cox·W·L + Cgdo
#   截止区 : Cgb = Cox·W·L
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AC Small-Signal Frequency Analysis")
print("=" * 60)

from spice_sim.analysis.dc_op import DCOperatingPoint
from spice_sim.analysis.ac_sweep import ACSweep

# ── Capacitance parameters (typical 1 µm CMOS technology) ─────────────────
# Cox ≈ 3.45 fF/µm²  (10 nm SiO2),  W=10 µm,  L=1 µm
AC_CGG  = 34.5e-15   # Cox·W·L  = 34.5 fF  (intrinsic gate cap)
AC_CGSO = 3.0e-15    # gate-source overlap  3 fF  (0.3 fF/µm · 10 µm)
AC_CGDO = 3.0e-15    # gate-drain  overlap  3 fF

# AC bias point: VIN = 1.5 V (deep saturation, good gain)
AC_VIN = 1.5

# NMOS model with capacitances (same DC params, add cgg/cgso/cgdo)
nmos_ac = NMOSModel(vth=VTH, kp=KP, lam=LAM, wl=WL, delta=DELTA,
                    cgg=AC_CGG, cgso=AC_CGSO, cgdo=AC_CGDO)

# Solve DC operating point at bias
dc_solver = DCOperatingPoint(nmos_ac, R_D, VDD)
dc_op = dc_solver.solve(AC_VIN)

print(f"\nDC bias at VIN = {AC_VIN} V:")
print(f"  VOUT = {dc_op['vout']:.4f} V")
print(f"  ID   = {dc_op['id']*1e3:.4f} mA")
print(f"  gm   = {dc_op['gm']*1e3:.4f} mA/V")
print(f"  ro   = {dc_op['ro']/1e3:.2f} kΩ")
print(f"  Av   = {dc_op['av']:.3f}   (= {20*np.log10(abs(dc_op['av'])):.2f} dB)")
print(f"  Region: {dc_op['region']}")

# ── AC sweep ───────────────────────────────────────────────────────────────
ac = ACSweep(nmos_ac, R_D, dc_op)

# Print capacitance values
print(f"\nGate capacitances at bias:")
print(f"  Cgs = {ac.cgs*1e15:.2f} fF   (2/3·Cgg + Cgso)")
print(f"  Cgd = {ac.cgd*1e15:.2f} fF   (Cgdo only, in saturation)")
print(f"  Cgb = {ac.cgb*1e15:.2f} fF   (≈ 0 in strong inversion)")

# Frequency sweep: 1 kHz → 100 GHz, 500 points log-spaced
freq_array = np.logspace(3, 11, 500)   # 1 kHz to 100 GHz
ac_result  = ac.run(freq_array)

# Metrics
fT        = ac.f_T()
f3db      = ac.f_3db(ac_result)
dc_gain_db = ac_result["H_db"][0]

print(f"\nAC metrics:")
print(f"  DC gain        |H(0)| = {dc_gain_db:.2f} dB  "
      f"(|Av| = {abs(dc_op['av']):.3f})")
if f3db is not None:
    print(f"  -3 dB bandwidth       = {f3db/1e6:.3f} MHz")
else:
    print("  -3 dB bandwidth       = > swept range")
if fT is not None:
    print(f"  Unity-gain freq  fT   = {fT/1e9:.3f} GHz  "
          f"[gm/(2π(Cgs+Cgd))]")

# ── Figure: Bode plot ──────────────────────────────────────────────────────
fig, (ax_mag, ax_phs) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(
    f"AC Frequency Response — CS NMOS Amplifier\n"
    f"Bias: VIN={AC_VIN}V  |Av|={abs(dc_op['av']):.2f}  "
    f"Cgg={AC_CGG*1e15:.0f}fF  Cgso=Cgdo={AC_CGSO*1e15:.0f}fF",
    fontsize=12)

freq_MHz = freq_array / 1e6

# Magnitude
ax_mag.semilogx(freq_MHz, ac_result["H_db"], color="#1f77b4", linewidth=2)
ax_mag.axhline(dc_gain_db,       color="gray",   linestyle=":", linewidth=1,
               label=f"DC gain = {dc_gain_db:.2f} dB")
ax_mag.axhline(dc_gain_db - 3.0, color="orange", linestyle="--", linewidth=1.2,
               label="-3 dB level")
if f3db is not None:
    ax_mag.axvline(f3db / 1e6, color="orange", linestyle="--", linewidth=1.2,
                   label=f"f_{{-3dB}} = {f3db/1e6:.2f} MHz")
    ax_mag.annotate(f"f_-3dB\n{f3db/1e6:.2f} MHz",
                    xy=(f3db/1e6, dc_gain_db - 3.0),
                    xytext=(f3db/1e6 * 3, dc_gain_db - 6),
                    arrowprops=dict(arrowstyle="->", color="orange"),
                    fontsize=9, color="orange")
if fT is not None:
    ax_mag.axvline(fT / 1e6, color="red", linestyle=":", linewidth=1.2,
                   label=f"fT = {fT/1e9:.2f} GHz")
    ax_mag.annotate(f"fT = {fT/1e9:.2f} GHz",
                    xy=(fT/1e6, 0.0),
                    xytext=(fT/1e6 * 0.3, -10),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=9, color="red")
ax_mag.set_ylabel("|H(f)| (dB)", fontsize=11)
ax_mag.set_ylim(bottom=min(ac_result["H_db"].min() - 3, dc_gain_db - 25))
ax_mag.legend(fontsize=9, loc="lower left")
ax_mag.grid(True, which="both", alpha=0.3)

# Phase
ax_phs.semilogx(freq_MHz, ac_result["phase_deg"], color="#d62728", linewidth=2)
ax_phs.axhline(-45, color="orange", linestyle="--", linewidth=1,
               alpha=0.7, label="-45° (pole frequency)")
ax_phs.axhline(-135, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax_phs.set_xlabel("Frequency (MHz)", fontsize=11)
ax_phs.set_ylabel("Phase (degrees)", fontsize=11)
ax_phs.legend(fontsize=9, loc="lower left")
ax_phs.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "ac_bode.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/ac_bode.png")

# ── Figure: Capacitance vs VIN ─────────────────────────────────────────────
# Show how Cgs/Cgd/Cgb vary with bias across the full sweep
cap_vin = np.linspace(0, 5, 300)
cap_cgs, cap_cgd, cap_cgb = [], [], []
for v in cap_vin:
    op = dc_solver.solve(v)
    c = nmos_ac.capacitances(op["vgs"], op["vds"], op["vsb"])
    cap_cgs.append(c[0])
    cap_cgd.append(c[1])
    cap_cgb.append(c[2])
cap_cgs = np.array(cap_cgs) * 1e15   # fF
cap_cgd = np.array(cap_cgd) * 1e15
cap_cgb = np.array(cap_cgb) * 1e15

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cap_vin, cap_cgs, color="#1f77b4", linewidth=2, label="Cgs")
ax.plot(cap_vin, cap_cgd, color="#d62728", linewidth=2, label="Cgd")
ax.plot(cap_vin, cap_cgb, color="#9467bd", linewidth=2, label="Cgb")
ax.axvline(VTH, color="gray", linestyle="--", linewidth=1, alpha=0.7,
           label=f"Vth={VTH}V")
ax.axvline(AC_VIN, color="green", linestyle=":", linewidth=1.5,
           label=f"AC bias ({AC_VIN}V)")
ax.set_xlabel("VIN (V)", fontsize=12)
ax.set_ylabel("Capacitance (fF)", fontsize=12)
ax.set_title("NMOS Gate Capacitances vs VIN  (Ch. 4 charge-based model)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "ac_caps_vs_vin.png"), dpi=150)
plt.close()
print("Saved: results/ac_caps_vs_vin.png")

# ── Updated overview: add AC row (row 4) ──────────────────────────────────
from matplotlib.gridspec import GridSpec as GS

fig2 = plt.figure(figsize=(18, 23))
fig2.suptitle(
    "CS NMOS Amplifier — Full Overview  (DC + secondary effects + AC)\n"
    f"VDD={VDD}V  RD={R_D/1e3:.0f}kΩ  Vth={VTH}V  "
    f"Kp={KP*1e6:.0f}µA/V²  λ={LAM}V⁻¹  W/L={WL:.0f}",
    fontsize=14, y=0.997)

gs2 = GS(5, 3, figure=fig2, hspace=0.50, wspace=0.35)

# Row 0 — VOUT + ID
a1 = fig2.add_subplot(gs2[0, :2])
a1.plot(data_full["vin"], data_full["vout"], color="#1f77b4", linewidth=2, label="Simulator")
a1.scatter(data_coarse["vin"], data_coarse["vout"], color="red", s=30, zorder=5, label="0.25V step")
a1.axvline(VTH, color="gray", linestyle="--", linewidth=1, alpha=0.7)
a1.axvline(vin_trans_pt, color="orange", linestyle="--", linewidth=1, alpha=0.8)
a1.set_xlabel("VIN (V)"); a1.set_ylabel("VOUT (V)")
a1.set_title("VOUT vs VIN (base model)"); a1.set_xlim(0, 5); a1.set_ylim(-0.1, VDD + 0.3)
a1.legend(fontsize=8); a1.grid(True, alpha=0.3)

a2 = fig2.add_subplot(gs2[0, 2])
a2.plot(data_full["vin"], data_full["id"] * 1e3, color="#2ca02c", linewidth=2)
a2.scatter(data_coarse["vin"], data_coarse["id"] * 1e3, color="red", s=20, zorder=5)
a2.axvline(VTH, color="gray", linestyle="--", linewidth=1, alpha=0.7)
a2.set_xlabel("VIN (V)"); a2.set_ylabel("ID (mA)")
a2.set_title("ID vs VIN"); a2.grid(True, alpha=0.3)

# Row 1 — zooms + Newton iters
a3 = fig2.add_subplot(gs2[1, 0])
mask_off = (data_full["vin"] >= 0.5) & (data_full["vin"] <= 1.2)
a3.plot(data_full["vin"][mask_off], data_full["vout"][mask_off], color="#1f77b4", linewidth=2)
a3.axvline(VTH, color="gray", linestyle="--", linewidth=1.5)
a3.set_xlabel("VIN (V)"); a3.set_ylabel("VOUT (V)")
a3.set_title(f"Zoom: Cutoff→Active (Vth={VTH}V)"); a3.grid(True, alpha=0.3)

a4 = fig2.add_subplot(gs2[1, 1])
margin = 0.5
mask_tr = ((data_full["vin"] >= vin_trans_pt - margin) &
           (data_full["vin"] <= vin_trans_pt + margin))
a4.plot(data_full["vin"][mask_tr], data_full["vout"][mask_tr], color="#1f77b4", linewidth=2)
a4.axvline(vin_trans_pt, color="orange", linestyle="--", linewidth=1.5)
a4.annotate(f"({vin_trans_pt:.2f}V,\n{vout_trans_pt:.3f}V)",
            xy=(vin_trans_pt, vout_trans_pt),
            xytext=(vin_trans_pt + 0.12, vout_trans_pt + 0.08),
            arrowprops=dict(arrowstyle="->", color="orange"), fontsize=8, color="orange")
a4.set_xlabel("VIN (V)"); a4.set_ylabel("VOUT (V)")
a4.set_title("Zoom: Lin/Sat Boundary"); a4.grid(True, alpha=0.3)

a5 = fig2.add_subplot(gs2[1, 2])
a5.bar(data_full["vin"], data_full["n_iter"], width=0.008, color="#17becf", alpha=0.8)
a5.set_xlabel("VIN (V)"); a5.set_ylabel("Iterations")
a5.set_title("Newton-Raphson Iterations"); a5.grid(True, alpha=0.3, axis="y")

# Row 2 — small-signal gm / ro / Av
a6 = fig2.add_subplot(gs2[2, 0])
a6.plot(data_full["vin"], data_full["gm"] * 1e3, color="#d62728", linewidth=2)
a6.set_xlabel("VIN (V)"); a6.set_ylabel("gm (mA/V)")
a6.set_title("Transconductance gm"); a6.grid(True, alpha=0.3)

a7 = fig2.add_subplot(gs2[2, 1])
ro_kOhm = np.minimum(data_full["ro"] / 1e3, 1e4)
valid    = ro_kOhm < 1e3
a7.plot(data_full["vin"], ro_kOhm, color="#9467bd", linewidth=2)
a7.set_ylim(0, ro_kOhm[valid].max() * 1.5 if valid.any() else 500)
a7.set_xlabel("VIN (V)"); a7.set_ylabel("ro (kΩ)")
a7.set_title("Output Resistance ro"); a7.grid(True, alpha=0.3)

a8 = fig2.add_subplot(gs2[2, 2])
a8.plot(data_full["vin"], np.abs(data_full["av"]), color="#8c564b", linewidth=2)
a8.set_xlabel("VIN (V)"); a8.set_ylabel("|Av|")
a8.set_title("Voltage Gain |Av| = gm·(ro||RD)"); a8.grid(True, alpha=0.3)

# Row 3 — Secondary effects comparison
a9 = fig2.add_subplot(gs2[3, :2])
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    a9.plot(data["vin"], data["vout"], color=c, linestyle=ls, linewidth=lw, label=label)
a9.axvline(VTH, color="gray", linestyle=":", linewidth=1, alpha=0.5)
a9.set_xlabel("VIN (V)"); a9.set_ylabel("VOUT (V)")
a9.set_title("Secondary Physical Effects: VOUT vs VIN")
a9.set_xlim(0, 5); a9.set_ylim(-0.1, VDD + 0.3)
a9.legend(fontsize=7, ncol=2, loc="upper right"); a9.grid(True, alpha=0.3)

a10 = fig2.add_subplot(gs2[3, 2])
for (label, data), c, ls, lw in zip(all_data.items(), colors, styles, lws):
    a10.plot(data["vin"], data["id"] * 1e3, color=c, linestyle=ls, linewidth=lw, label=label)
a10.set_xlabel("VIN (V)"); a10.set_ylabel("ID (mA)")
a10.set_title("Secondary Effects: ID vs VIN")
a10.legend(fontsize=6); a10.grid(True, alpha=0.3)

# Row 4 — AC Bode plot (magnitude + phase + capacitances)
a11 = fig2.add_subplot(gs2[4, :2])
a11.semilogx(freq_MHz, ac_result["H_db"], color="#1f77b4", linewidth=2)
a11.axhline(dc_gain_db,       color="gray",   linestyle=":",  linewidth=1,
            label=f"DC gain {dc_gain_db:.1f} dB")
a11.axhline(dc_gain_db - 3.0, color="orange", linestyle="--", linewidth=1.2,
            label="-3 dB")
if f3db is not None:
    a11.axvline(f3db / 1e6, color="orange", linestyle="--", linewidth=1.2,
                label=f"f_{{-3dB}}={f3db/1e6:.1f} MHz")
if fT is not None:
    a11.axvline(fT / 1e6, color="red", linestyle=":", linewidth=1.2,
                label=f"fT={fT/1e9:.2f} GHz")
a11.set_xlabel("Frequency (MHz)", fontsize=9)
a11.set_ylabel("|H| (dB)", fontsize=9)
a11.set_title(f"AC Bode Magnitude  (bias VIN={AC_VIN}V)", fontsize=10)
a11.legend(fontsize=7, loc="lower left")
a11.grid(True, which="both", alpha=0.3)

a12 = fig2.add_subplot(gs2[4, 2])
a12.plot(cap_vin, cap_cgs, color="#1f77b4", linewidth=1.8, label="Cgs")
a12.plot(cap_vin, cap_cgd, color="#d62728", linewidth=1.8, label="Cgd")
a12.plot(cap_vin, cap_cgb, color="#9467bd", linewidth=1.8, label="Cgb")
a12.axvline(VTH,    color="gray",  linestyle="--", linewidth=1, alpha=0.7)
a12.axvline(AC_VIN, color="green", linestyle=":",  linewidth=1.5, label=f"bias {AC_VIN}V")
a12.set_xlabel("VIN (V)", fontsize=9); a12.set_ylabel("Cap (fF)", fontsize=9)
a12.set_title("Gate Caps vs VIN (Ch. 4)", fontsize=10)
a12.legend(fontsize=7); a12.grid(True, alpha=0.3)

fig2.savefig(os.path.join(RESULTS_DIR, "overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/overview.png  (5-row: DC + zooms + small-signal + secondary effects + AC)")
print("\nDone.")
