"""
改进节点分析法（MNA）—— 扩展版，支持源/漏极寄生串联电阻（Rs / Rd）。

节点编号（固定）：
  0 : GND（参考，不入方程）
  1 : Gate   — 由电压源 VIN 约束
  2 : 外部 Drain / VOUT
  3 : VDD    — 由电压源 VDD 约束
  4 : 本征 Source S'（Rs 在 S' 和 GND 之间；Rs=0 时约束 V4=0）
  5 : 本征 Drain  D'（Rd 在 VOUT 和 D' 之间；Rd=0 时约束 V5=V2）

未知量向量 x = [V1, V2, V3, V4, V5, I_VIN, I_VDD]  (N=7)

元件连接：
  RD_load  : 节点3 → 节点2（负载电阻）
  Rd_para  : 节点2 → 节点5（漏极寄生串联电阻，Rd=0 时短路）
  Rs_para  : 节点4 → 节点0（源极寄生串联电阻，Rs=0 时短路）
  NMOS M0  : Gate=1, Drain=5, Source=4, Bulk=0
    VGS_int = V1 - V4
    VDS_int = V5 - V4
    VSB     = V4

当 Rs=Rd=0 时，约束 V4=0 / V5=V2 使系统退化为等价于原始 5 变量结果。
"""

import numpy as np


class MNABuilder:
    # 未知量索引
    IDX_V1   = 0
    IDX_V2   = 1
    IDX_V3   = 2
    IDX_V4   = 3   # 本征 Source
    IDX_V5   = 4   # 本征 Drain
    IDX_IVIN = 5
    IDX_IVDD = 6
    N        = 7

    def __init__(self, R_load, nmos_model, Rs=0.0, Rd=0.0):
        """
        Parameters
        ----------
        R_load     : 负载电阻 RD (Ω)
        nmos_model : NMOSModel 实例
        Rs         : 源极寄生串联电阻 (Ω)，默认 0
        Rd         : 漏极寄生串联电阻 (Ω)，默认 0
        """
        self.G_load = 1.0 / R_load
        self.mos    = nmos_model
        self.Rs     = float(Rs)
        self.Rd     = float(Rd)
        self.G_rs   = 1.0 / Rs if Rs > 0.0 else 0.0
        self.G_rd   = 1.0 / Rd if Rd > 0.0 else 0.0

    def _mos_stamp(self, x):
        """计算当前工作点的 NMOS 线性化参数。"""
        V1, V4, V5 = x[self.IDX_V1], x[self.IDX_V4], x[self.IDX_V5]
        vgs_int = V1 - V4
        vds_int = V5 - V4
        vsb     = V4
        ID_k, gm_k, gds_k, gmb_k = self.mos._eval(vgs_int, vds_int, vsb)
        # 伴随电流源常数项（线性化展开的截距）
        I_eq = ID_k - gm_k * vgs_int - gds_k * vds_int - gmb_k * vsb
        return ID_k, gm_k, gds_k, gmb_k, I_eq

    def build(self, x, vin, vdd):
        """
        在工作点 x 处构建线性化 MNA 方程组 A·x_new = b。

        参数
        ----
        x   : 当前迭代解向量（长度 7）
        vin : 已知 VIN
        vdd : 已知 VDD

        返回
        ----
        A (7×7), b (7,)
        """
        _, gm_k, gds_k, gmb_k, I_eq = self._mos_stamp(x)

        A = np.zeros((self.N, self.N))
        b = np.zeros(self.N)

        # ── 行0：节点1 KCL（Gate，无电流分支） ───────────────────────────────
        A[0, self.IDX_IVIN] = 1.0

        # ── 行1：节点2 KCL（外部 VOUT） ──────────────────────────────────────
        # RD_load 贡献（节点3→节点2）
        A[1, self.IDX_V2] -= self.G_load
        A[1, self.IDX_V3] += self.G_load

        if self.G_rd > 0.0:
            # Rd_para 贡献（节点2→节点5），NMOS 电流不在此行
            A[1, self.IDX_V2] -= self.G_rd
            A[1, self.IDX_V5] += self.G_rd
        else:
            # Rd=0 → V5=V2，NMOS 漏极直接连节点2，ID 在本行 stamp
            # (V3-V2)*G_load - ID = 0
            # ID ≈ I_eq + gm*V1 + (-gm-gds+gmb)*V4 + gds*V2
            A[1, self.IDX_V1] -= gm_k
            A[1, self.IDX_V4] += gm_k + gds_k - gmb_k
            A[1, self.IDX_V2] -= gds_k
            b[1]               = I_eq

        # ── 行2：节点3 KCL（VDD） ─────────────────────────────────────────────
        A[2, self.IDX_V2]   += self.G_load
        A[2, self.IDX_V3]   -= self.G_load
        A[2, self.IDX_IVDD] += 1.0

        # ── 行3：节点4（本征 Source S'）或约束 V4=0 ──────────────────────────
        if self.G_rs > 0.0:
            # KCL: ID - V4·G_rs = 0
            # gm·V1 + (-gm-gds+gmb-G_rs)·V4 + gds·V5 = -I_eq
            A[3, self.IDX_V1] += gm_k
            A[3, self.IDX_V4] += -gm_k - gds_k + gmb_k - self.G_rs
            A[3, self.IDX_V5] += gds_k
            b[3]               = -I_eq
        else:
            # Rs=0 → 约束 V4 = 0
            A[3, self.IDX_V4] = 1.0
            b[3]               = 0.0

        # ── 行4：节点5（本征 Drain D'）或约束 V5=V2 ──────────────────────────
        if self.G_rd > 0.0:
            # KCL: (V2-V5)·G_rd - ID = 0
            # -gm·V1 + (gm+gds-gmb)·V4 + G_rd·V2 - (G_rd+gds)·V5 = I_eq
            A[4, self.IDX_V1] -= gm_k
            A[4, self.IDX_V4] += gm_k + gds_k - gmb_k
            A[4, self.IDX_V2] += self.G_rd
            A[4, self.IDX_V5] -= self.G_rd + gds_k
            b[4]               = I_eq
        else:
            # Rd=0 → 约束 V5 = V2
            A[4, self.IDX_V5] =  1.0
            A[4, self.IDX_V2] = -1.0
            b[4]               = 0.0

        # ── 行5：KVL VIN ──────────────────────────────────────────────────────
        A[5, self.IDX_V1] = 1.0
        b[5]               = vin

        # ── 行6：KVL VDD ──────────────────────────────────────────────────────
        A[6, self.IDX_V3] = 1.0
        b[6]               = vdd

        return A, b

    def residual(self, x, vin, vdd):
        """非线性残差向量 F(x)，F=0 即精确解。"""
        V1, V2, V3, V4, V5 = (x[self.IDX_V1], x[self.IDX_V2], x[self.IDX_V3],
                               x[self.IDX_V4], x[self.IDX_V5])
        I_VIN, I_VDD = x[self.IDX_IVIN], x[self.IDX_IVDD]

        vgs_int = V1 - V4
        vds_int = V5 - V4
        vsb     = V4
        ID = self.mos.ids(vgs_int, vds_int, vsb)

        F = np.zeros(self.N)
        F[0] = I_VIN                                        # 节点1 KCL
        F[1] = (V3 - V2) * self.G_load - (V2 - V5) * self.G_rd  # 节点2 KCL
        if self.G_rd == 0.0:
            # Rd=0 时节点2直接连 NMOS 漏极
            F[1] = (V3 - V2) * self.G_load - ID
        F[2] = -(V3 - V2) * self.G_load + I_VDD            # 节点3 KCL
        if self.G_rs > 0.0:
            F[3] = ID - V4 * self.G_rs                     # 节点4 KCL
        else:
            F[3] = V4                                       # 约束 V4=0
        if self.G_rd > 0.0:
            F[4] = (V2 - V5) * self.G_rd - ID              # 节点5 KCL
        else:
            F[4] = V5 - V2                                  # 约束 V5=V2
        F[5] = V1 - vin                                     # KVL VIN
        F[6] = V3 - vdd                                     # KVL VDD

        return F
