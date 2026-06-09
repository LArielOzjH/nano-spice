# nano-spice

A from-scratch SPICE DC simulator in Python, built as a course project for *Device Models and SPICE Simulation* (Fudan University, 2026).

Simulates a common-source NMOS amplifier using **Modified Nodal Analysis (MNA)** + **Newton-Raphson** iteration, with a progressively enhanced MOSFET model from Level-1 up to Level-2 physical effects.

---

## Circuit

```
VDD (5V)
  |
 [RD = 10kΩ]
  |
  +-----> VOUT
  |
 [M0] NMOS  Gate <- VIN
  |         Vth=0.8V, Kp=100uA/V^2, lambda=0.02V^-1, W/L=10
 GND
```

---

## Features

### Level-1 Model (Shichman-Hodges)
- Cutoff / Linear / Saturation regions with channel-length modulation (λ)
- Smooth linear↔saturation transition via **softmin** function (avoids Newton-Raphson gradient discontinuity)
- Analytic `gm` and `gds` for quadratic convergence

### Level-2 Model (Higher-Order Physical Effects)

| Effect | Parameter | Physical Origin |
|--------|-----------|-----------------|
| **2A Subthreshold conduction** | `n_sub` | Unified `Vgsteff = n·VT·ln(1+exp(Vov/nVT))` covers weak inversion continuously |
| **2B Mobility degradation** | `theta` | High transverse field → surface scattering → `μeff = μ0/(1+θ·Vov)` |
| **2C DIBL** | `eta` | Drain field lowers threshold: `Vth_eff = Vth0 − η·VDS` |
| **2D Body effect** | `gamma`, `phi_f` | Source-bulk bias raises threshold: `Vth += γ(√(VSB+2φF)−√(2φF))` |
| **2E Parasitic Rs/Rd** | `Rs`, `Rd` | Source/drain series resistance — adds two intrinsic nodes to MNA |

All Level-2 parameters default to zero (no effect), so the model degrades exactly to Level-1.

### Solver
- **MNA** with 7 unknowns: `[V1, V2, V3, V4(S'), V5(D'), I_VIN, I_VDD]`
- **Newton-Raphson** with adaptive damping (residual-based λ-halving)
- Analytic Jacobian including `gm`, `gds`, `gmb` — typically converges in 2–6 iterations

### Analysis
- DC operating point sweep: VIN = 0→5V, step 0.25V (coarse) + 2–5mV near critical transitions
- Small-signal parameters at each point: `gm`, `ro`, `|Av| = gm·(ro‖RD)`
- Level-1 vs Level-2 comparison: effect of each physical enhancement on VOUT and ID

### Level-3 Model — AC Small-Signal Frequency Sweep

Extends MNA to the complex frequency domain:

| Feature | Details |
|---------|---------|
| **Gate capacitance model** (Ch. 4) | Cgs/Cgd/Cgb computed from charge-based partition with smooth sigmoid region transitions |
| **Region-dependent partition** | Cutoff: Cgb=Cgg; Linear: Cgs=Cgd=Cgg/2; Saturation: Cgs=2/3·Cgg, Cgd≈0 |
| **Overlap capacitances** | Cgso, Cgdo added to intrinsic caps (technology parameters) |
| **Complex MNA** | Y(jω) = G + jω·C, solved at each frequency point |
| **Transfer function** | H(jω) = VOUT/VIN computed from unit AC excitation |
| **−3 dB bandwidth** | Interpolated from |H(jω)| roll-off |
| **fT** | Unity current-gain frequency = gm / (2π·(Cgs+Cgd)) |

---

## Project Structure

```
nano-spice/
├── simulate.py                  # Main script — run this
├── spice_sim/
│   ├── models/
│   │   ├── nmos.py              # NMOS Level-1/2/3 model + caps + derivatives
│   │   └── resistor.py          # Linear resistor + MNA stamp
│   ├── core/
│   │   ├── mna.py               # MNA matrix builder (7-variable, Rs/Rd support)
│   │   └── newton.py            # Newton-Raphson solver with damping
│   └── analysis/
│       ├── dc_op.py             # Single-point DC operating point
│       ├── dc_sweep.py          # DC parameter sweep
│       └── ac_sweep.py          # AC small-signal frequency sweep (Level-3)
└── results/                     # Generated figures (git-ignored)
```

---

## Quick Start

```bash
# No installation needed beyond numpy + matplotlib
pip install numpy matplotlib

python simulate.py
```

Output figures are saved to `results/`:

| File | Content |
|------|---------|
| `vout_vs_vin.png` | VOUT vs VIN — coarse samples + continuous curve |
| `vout_vs_vin_zoom.png` | Zoomed views near Vth and Lin/Sat boundary |
| `id_vs_vin.png` | Drain current ID vs VIN |
| `smallsig_vs_vin.png` | gm / ro / \|Av\| vs VIN |
| `newton_iters.png` | Newton-Raphson iteration count per point |
| `level2_comparison.png` | Level-1 vs each Level-2 effect on VOUT |
| `level2_gm_id.png` | Level-1 vs Level-2 on gm and ID |
| `ac_bode.png` | AC Bode plot: magnitude + phase, with −3dB and fT markers |
| `ac_caps_vs_vin.png` | Cgs/Cgd/Cgb vs VIN (Ch. 4 charge-based model) |
| `overview.png` | All plots in one image (5×3 grid, including AC row) |

---

## Using the Model

```python
from spice_sim.models.nmos import NMOSModel
from spice_sim.analysis.dc_sweep import DCSweep
import numpy as np

# Level-1 (default)
nmos = NMOSModel(vth=0.8, kp=100e-6, lam=0.02, wl=10)

# Level-2 (all effects on)
nmos_l2 = NMOSModel(
    vth=0.8, kp=100e-6, lam=0.02, wl=10,
    theta=0.10,   # mobility degradation
    eta=0.03,     # DIBL
    gamma=0.50,   # body effect
    phi_f=0.70,   # 2*phi_F surface potential
    n_sub=1.30,   # subthreshold slope factor
)

sweep = DCSweep(nmos_l2, R_d=10e3, vdd=5.0, Rs=50, Rd=50)
results = sweep.run(np.arange(0, 5.01, 0.25), verbose=True)
```

---

## Theory Notes

### MNA Formulation

For the common-source circuit, the 7-variable MNA system is:

```
x = [V1, V2, V3, V4(S'), V5(D'), I_VIN, I_VDD]

Node 1 KCL  : I_VIN = 0
Node 2 KCL  : (V3-V2)*G_RD - (V2-V5)*G_rd = 0
Node 3 KCL  : -(V3-V2)*G_RD + I_VDD = 0
Node 4 KCL  : ID(VGS_int, VDS_int, VSB) - V4*G_rs = 0
Node 5 KCL  : (V2-V5)*G_rd - ID(...) = 0
KVL VIN     : V1 = VIN
KVL VDD     : V3 = VDD
```

The NMOS is linearized at each Newton step:

```
ID ≈ Ieq + gm*V1 + (-gm - gds + gmb)*V4 + gds*V5
```

### Smooth Transition

Instead of a hard `if vds < vov` branch, a differentiable **softmin** is used:

```
Vds_eff = softmin(vds, Vgsteff)
        = 0.5 * (vds + Vgsteff - sqrt((vds - Vgsteff)^2 + delta^2))
```

This keeps the Jacobian continuous and preserves Newton-Raphson quadratic convergence near the Linear/Saturation boundary.

---

## Course Context

**器件模型与 SPICE 仿真 / Device Models and SPICE Simulation**  
Fudan University · School of Microelectronics · Spring 2026

Key references from course lectures:
- Ch. 9: MNA circuit equation construction
- Ch. 10: LU decomposition for linear systems
- Ch. 11: Newton-Raphson for nonlinear DC analysis
- Ch. 2–3: BSIM MOSFET models (Vgsteff, Abul, mobility)
- Ch. 4: Charge and capacitance model — Cgs/Cgd/Cgb partitioning
- Ch. 5: Parasitic gate/source/drain resistances
