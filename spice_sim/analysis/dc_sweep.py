"""
DC 参数扫描分析。

对 VIN 在指定范围内逐步求解 DC 工作点，返回完整扫描结果。
"""

import numpy as np
from spice_sim.analysis.dc_op import DCOperatingPoint


class DCSweep:
    """
    DC 扫描分析器。

    参数
    ----
    nmos_model : NMOSModel 实例
    R_d        : 漏极负载电阻 (Ω)
    vdd        : 电源电压 (V)
    """

    def __init__(self, nmos_model, R_d, vdd, Rs=0.0, Rd=0.0):
        self.op_solver = DCOperatingPoint(nmos_model, R_d, vdd, Rs=Rs, Rd=Rd)

    def run(self, vin_array, verbose=False):
        """
        对 vin_array 中每个 VIN 值求解工作点。

        参数
        ----
        vin_array : VIN 值的数组（V）
        verbose   : 是否打印每步结果

        返回
        ----
        results : list of dict（每个 VIN 对应 dc_op.solve 的返回值）
        """
        results = []
        for vin in vin_array:
            res = self.op_solver.solve(vin)
            results.append(res)
            if verbose:
                status = "OK" if res["converged"] else "FAIL"
                print(f"  VIN={vin:5.3f}V  VOUT={res['vout']:7.4f}V  "
                      f"ID={res['id']*1e3:7.4f}mA  "
                      f"Region={res['region']:11s}  "
                      f"Iter={res['n_iter']:2d}  [{status}]")
        return results

    @staticmethod
    def to_arrays(results):
        """将 results list 解包为各物理量的 numpy 数组，方便绘图。"""
        keys = ["vin", "vout", "id", "vgs", "vds", "vsb",
                "gm", "gds", "gmb", "ro", "av", "n_iter"]
        return {k: np.array([r[k] for r in results]) for k in keys}
