"""
AC small-signal frequency sweep.

Extends DC MNA to the complex domain:
    Y(jω) = G + jω·C

where G is the linearised conductance matrix evaluated at the DC operating
point and C is assembled from the NMOS gate-capacitance model (Chapter 4).

The transfer function H(jω) = VOUT(jω) / VIN(jω) is obtained by solving
    Y(jω) · x = b_ac
with a unit AC input (b_ac[5]=1, all others 0) at each frequency.

Key quantities
--------------
  -3dB bandwidth : frequency where |H| drops 3 dB from DC value
  fT             : unity current-gain frequency = gm / (2π·(Cgs+Cgd))
"""

import numpy as np
from spice_sim.core.mna import MNABuilder


class ACSweep:
    """
    AC small-signal frequency sweep for the CS NMOS amplifier.

    Parameters
    ----------
    nmos_model : NMOSModel  (cgg, cgso, cgdo must be set for cap model)
    R_load     : load resistance RD (Ω)
    dc_result  : dict returned by DCOperatingPoint.solve()
    Rs, Rd     : parasitic series resistances — must match those used for DC
    """

    def __init__(self, nmos_model, R_load, dc_result, Rs=0.0, Rd=0.0):
        self.mos    = nmos_model
        self.R_load = float(R_load)
        self.Rs     = float(Rs)
        self.Rd     = float(Rd)

        # DC operating-point small-signal parameters
        self.gm_dc  = dc_result["gm"]
        self.gds_dc = dc_result["gds"]
        self.gmb_dc = dc_result["gmb"]
        self.vgs_dc = dc_result["vgs"]
        self.vds_dc = dc_result["vds"]
        self.vsb_dc = dc_result["vsb"]
        self.dc_gain = dc_result["av"]   # signed voltage gain

        # Gate capacitances at DC OP
        self.cgs, self.cgd, self.cgb = nmos_model.capacitances(
            self.vgs_dc, self.vds_dc, self.vsb_dc
        )

        # Build G and C matrices once (they don't depend on frequency)
        self.G, self.C = self._build_matrices()

    # ── Matrix assembly ────────────────────────────────────────────────────

    def _build_matrices(self):
        """
        Construct 7×7 conductance (G) and capacitance (C) matrices.

        G replicates the MNABuilder linearised conductance matrix at the DC OP.
        C stamps Cgs / Cgd / Cgb from the gate-capacitance model.
        """
        N = MNABuilder.N
        G = np.zeros((N, N))
        C = np.zeros((N, N))

        V1, V2, V3, V4, V5 = 0, 1, 2, 3, 4
        IVIN, IVDD = 5, 6

        G_load = 1.0 / self.R_load
        G_rs   = 1.0 / self.Rs if self.Rs > 0.0 else 0.0
        G_rd   = 1.0 / self.Rd if self.Rd > 0.0 else 0.0
        gm, gds, gmb = self.gm_dc, self.gds_dc, self.gmb_dc

        # ── Conductance matrix (mirrors MNABuilder.build) ──────────────────
        # Row 0: node-1 KCL (gate)
        G[0, IVIN] = 1.0

        # Row 1: node-2 KCL (VOUT / external drain)
        G[1, V2] -= G_load
        G[1, V3] += G_load
        if G_rd > 0.0:
            G[1, V2] -= G_rd
            G[1, V5] += G_rd
        else:
            # Rd=0 → V5=V2; stamp linearised NMOS current on this row
            G[1, V1] -= gm
            G[1, V4] += gm + gds - gmb
            G[1, V2] -= gds

        # Row 2: node-3 KCL (VDD)
        G[2, V2]   += G_load
        G[2, V3]   -= G_load
        G[2, IVDD] += 1.0

        # Row 3: node-4 KCL (intrinsic source S') or constraint V4=0
        if G_rs > 0.0:
            G[3, V1] += gm
            G[3, V4] += -gm - gds + gmb - G_rs
            G[3, V5] += gds
        else:
            G[3, V4] = 1.0          # constraint V4 = 0

        # Row 4: node-5 KCL (intrinsic drain D') or constraint V5=V2
        if G_rd > 0.0:
            G[4, V1] -= gm
            G[4, V4] += gm + gds - gmb
            G[4, V2] += G_rd
            G[4, V5] -= G_rd + gds
        else:
            G[4, V5] =  1.0         # constraint V5 = V2
            G[4, V2] = -1.0

        # Row 5: KVL VIN
        G[5, V1] = 1.0

        # Row 6: KVL VDD
        G[6, V3] = 1.0

        # ── Capacitance matrix ─────────────────────────────────────────────
        # Only KCL rows (0-4) are stamped; KVL rows (5-6) are never modified.
        cgs, cgd, cgb = self.cgs, self.cgd, self.cgb

        # Cgb: gate (V1) to GND (bulk) — only diagonal of row 0
        C[V1, V1] += cgb

        # Cgs: gate (V1) to source node
        C[V1, V1] += cgs
        if G_rs > 0.0:
            # V4 is a real KCL node
            C[V1, V4] -= cgs
            C[V4, V1] -= cgs
            C[V4, V4] += cgs
        # else: Rs=0 → source is GND (row 3 is constraint), no extra stamp

        # Cgd: gate (V1) to drain node
        C[V1, V1] += cgd
        if G_rd > 0.0:
            # V5 is a real KCL node
            C[V1, V5] -= cgd
            C[V5, V1] -= cgd
            C[V5, V5] += cgd
        else:
            # Rd=0 → V5=V2; Cgd connects V1 to V2 (Miller capacitance path)
            C[V1, V2] -= cgd
            C[V2, V1] -= cgd
            C[V2, V2] += cgd

        return G, C

    # ── Frequency sweep ────────────────────────────────────────────────────

    def run(self, freq_array):
        """
        Compute AC transfer function H(jω) = VOUT / VIN at each frequency.

        Parameters
        ----------
        freq_array : 1-D array of frequencies (Hz)

        Returns
        -------
        dict with keys:
            'freq'      : input freq_array (Hz)
            'H'         : complex transfer function array
            'H_db'      : 20·log10|H| (dB)
            'phase_deg' : phase in degrees
        """
        freqs = np.asarray(freq_array, dtype=float)
        H     = np.empty(len(freqs), dtype=complex)

        # Unit AC excitation: V1=1V (signal), V3=0V (VDD is AC ground)
        b_ac = np.zeros(MNABuilder.N)
        b_ac[5] = 1.0   # KVL row for VIN: V1 = 1 V
        # b_ac[6] stays 0: KVL row for VDD: V3 = 0 V (AC short)

        for i, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            Y = self.G + (1j * omega) * self.C
            try:
                x_ac = np.linalg.solve(Y, b_ac)
                H[i] = x_ac[MNABuilder.IDX_V2]   # VOUT phasor (VIN=1)
            except np.linalg.LinAlgError:
                H[i] = 0.0 + 0.0j

        H_db      = 20.0 * np.log10(np.abs(H) + 1e-30)
        phase_deg = np.degrees(np.angle(H))

        return {
            "freq":      freqs,
            "H":         H,
            "H_db":      H_db,
            "phase_deg": phase_deg,
        }

    # ── Derived metrics ────────────────────────────────────────────────────

    def f_3db(self, result):
        """
        Find -3 dB bandwidth from a sweep result dict.

        Locates the first frequency where |H| drops 3 dB below the DC value
        (first point), using linear interpolation between adjacent samples.
        Returns None if the gain never drops 3 dB in the swept range.
        """
        H_db  = result["H_db"]
        freqs = result["freq"]
        dc_db = H_db[0]
        target = dc_db - 3.0

        for i in range(1, len(H_db)):
            if H_db[i] <= target:
                # Linear interpolation in log-frequency space
                lf1 = np.log10(freqs[i - 1])
                lf2 = np.log10(freqs[i])
                h1, h2 = H_db[i - 1], H_db[i]
                t = (target - h1) / (h2 - h1)
                return 10.0 ** (lf1 + t * (lf2 - lf1))
        return None

    def f_T(self):
        """
        Unity current-gain frequency fT = gm / (2π·(Cgs + Cgd)).

        This is the intrinsic transistor speed limit independent of load.
        Returns None if the capacitance model is disabled (cgg=None).
        """
        total_cap = self.cgs + self.cgd
        if total_cap < 1e-30:
            return None
        return self.gm_dc / (2.0 * np.pi * total_cap)
