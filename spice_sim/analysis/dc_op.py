"""
单点 DC 工作点分析（适配 7 变量 MNA）。
"""

import numpy as np
from spice_sim.core.mna import MNABuilder
from spice_sim.core.newton import NewtonSolver


class DCOperatingPoint:
    """
    DC 工作点求解器。

    Parameters
    ----------
    nmos_model : NMOSModel 实例
    R_load     : 负载电阻 RD (Ω)
    vdd        : 电源电压 (V)
    Rs         : 源极寄生串联电阻 (Ω)，默认 0
    Rd         : 漏极寄生串联电阻 (Ω)，默认 0
    """

    def __init__(self, nmos_model, R_load, vdd, Rs=0.0, Rd=0.0):
        self.mos   = nmos_model
        self.R_load = R_load
        self.vdd   = vdd
        self.mna   = MNABuilder(R_load, nmos_model, Rs=Rs, Rd=Rd)
        self.solver = NewtonSolver()

    def solve(self, vin):
        """
        求解 VIN 对应的 DC 工作点。

        Returns
        -------
        dict with keys: vin, vout, id, vgs, vds, vsb,
                        gm, gds, gmb, ro, av, region, n_iter, converged
        """
        # 初始猜测：V1=VIN, V2=VDD/2, V3=VDD, V4=0, V5=VDD/2, I=0
        x0 = np.array([vin, self.vdd / 2.0, self.vdd,
                        0.0, self.vdd / 2.0, 0.0, 0.0])

        x, n_iter, converged, _ = self.solver.solve(
            self.mna, x0, vin, self.vdd
        )

        V4  = x[MNABuilder.IDX_V4]
        V5  = x[MNABuilder.IDX_V5]
        vout = x[MNABuilder.IDX_V2]   # 外部 VOUT（始终是报告用的输出电压）

        vgs_int = x[MNABuilder.IDX_V1] - V4
        vds_int = V5 - V4
        vsb     = V4

        _, gm_val, gds_val, gmb_val = self.mos._eval(vgs_int, vds_int, vsb)
        id_val  = self.mos.ids(vgs_int, vds_int, vsb)
        ro      = 1.0 / gds_val if gds_val > 1e-15 else 1e15

        ro_par_RL = (ro * self.R_load) / (ro + self.R_load)
        av = -gm_val * ro_par_RL

        return {
            "vin":       vin,
            "vout":      vout,
            "id":        id_val,
            "vgs":       vgs_int,
            "vds":       vds_int,
            "vsb":       vsb,
            "gm":        gm_val,
            "gds":       gds_val,
            "gmb":       gmb_val,
            "ro":        ro,
            "av":        av,
            "region":    self.mos.region(vgs_int, vds_int, vsb),
            "n_iter":    n_iter,
            "converged": converged,
        }
