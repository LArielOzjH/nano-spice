"""
NMOS 器件模型（Level-2，含高阶效应）

在 Level-1 Shichman-Hodges 模型基础上增加：
  2A. 亚阈值导通  — 通过统一有效栅压 Vgsteff 连续覆盖截止/弱反型/强反型
  2B. 迁移率衰减  — 横向强电场下载流子散射增强，μeff = μ0/(1+θ·Vov)
  2C. DIBL 效应   — 高漏压降低阈值电压，Vth_eff = Vth0 - η·VDS
  2D. 体效应      — 源-衬底偏压升高阈值，Vth_eff += γ(√(VSB+2φF) - √(2φF))

所有效应参数均有"关闭默认值"，令 theta=eta=gamma=0，n_sub=None 时
模型退化为与原 Level-1 行为完全一致的结果。

所有导数（gm, gds, gmb）均为解析推导，确保 Newton-Raphson 二次收敛。
"""

import numpy as np

_VT_ROOM = 0.025852  # kT/q at 300 K (V)


class NMOSModel:
    """
    NMOS Level-2 器件模型。

    基础参数 (Level-1)
    --------
    vth   : 零偏阈值电压 Vth0 (V)
    kp    : 工艺跨导 μ0·Cox (A/V²)
    lam   : 沟道长度调制系数 λ (V⁻¹)
    wl    : W/L 比
    delta : softmin 平滑参数 (V)

    Level-2 高阶参数
    ----------------
    theta  : 迁移率衰减系数 θ (V⁻¹)，典型值 0.05~0.5，默认 0（关闭）
    eta    : DIBL 系数 η (V/V)，典型值 0.01~0.1，默认 0（关闭）
    gamma  : 体效应系数 γ (V^0.5)，典型值 0.3~1.0，默认 0（关闭）
    phi_f  : 表面势 2φF (V)，典型值 0.6~0.8，默认 0.7
    n_sub  : 亚阈值斜率因子 n，典型值 1.1~2.0；
             若为 None（默认）则关闭亚阈值导通，VGS≤Vth 时 ID=0
    """

    def __init__(self, vth=0.8, kp=100e-6, lam=0.02, wl=10.0, delta=0.02,
                 theta=0.0, eta=0.0, gamma=0.0, phi_f=0.7, n_sub=None,
                 cgg=None, cgso=0.0, cgdo=0.0,
                 IS=0.0):
        self.vth   = vth
        self.kp    = kp
        self.lam   = lam
        self.wl    = wl
        self.delta = delta
        self.theta = theta
        self.eta   = eta
        self.gamma = gamma
        self.phi_f = phi_f
        self.n_sub = n_sub
        # Capacitance model (Chapter 4)
        # cgg  : total intrinsic gate cap Cox·W·L (F); None disables cap model
        # cgso : gate-source overlap cap (F)
        # cgdo : gate-drain overlap cap (F)
        self.cgg  = cgg
        self.cgso = float(cgso)
        self.cgdo = float(cgdo)
        # BD/BS junction diode saturation current (A)
        # IS=0 disables diodes (default, backward-compatible)
        self.IS = float(IS)

    # ── 内部：softmin 及其偏导 ────────────────────────────────────────────────

    def _softmin(self, a, b):
        d = a - b
        return 0.5 * (a + b - np.sqrt(d**2 + self.delta**2))

    def _softmin_da(self, a, b):
        d = a - b
        return 0.5 * (1.0 - d / np.sqrt(d**2 + self.delta**2))

    def _softmin_db(self, a, b):
        return 1.0 - self._softmin_da(a, b)

    # ── 内部：计算所有中间量 ──────────────────────────────────────────────────

    def _eval(self, vgs, vds, vsb=0.0):
        """
        在工作点 (vgs, vds, vsb) 处计算 ID 及其三个偏导数。

        返回
        ----
        (ID, gm, gds, gmb)
        """
        # ── 2D. 体效应 ───────────────────────────────────────────────────────
        # dvth_body/dVsb = γ / (2·√(|vsb|+2φF))
        vsb_abs = abs(vsb)
        if self.gamma > 0.0:
            phi_arg = vsb_abs + self.phi_f      # = |VSB| + 2φF (用 phi_f 表示 2φF)
            vth_body    = self.gamma * (np.sqrt(phi_arg) - np.sqrt(self.phi_f))
            dvth_dvsb   = self.gamma * 0.5 / np.sqrt(phi_arg)  # ≥ 0
        else:
            vth_body  = 0.0
            dvth_dvsb = 0.0

        # ── 2C. DIBL ──────────────────────────────────────────────────────────
        vth_dibl = -self.eta * vds   # dVth/dVds = -η → dVov/dVds = +η

        # 有效阈值和过驱动电压
        vth_eff = self.vth + vth_body + vth_dibl
        vov     = vgs - vth_eff

        # ── 2A. 统一 Vgsteff（含亚阈值过渡） ────────────────────────────────
        if self.n_sub is not None:
            nVT = self.n_sub * _VT_ROOM
            arg = vov / nVT
            # 数值稳定的 log1p(exp(x))
            if arg > 50.0:
                vgsteff = vov           # 强反型，sigmoid ≈ 1
                s        = 1.0
            elif arg < -50.0:
                vgsteff = nVT * np.exp(arg)  # 深亚阈值
                s        = np.exp(arg)        # ≈ 0
            else:
                vgsteff = nVT * np.log1p(np.exp(arg))
                e        = np.exp(arg)
                s        = e / (1.0 + e)     # sigmoid = dVgsteff/dVov
        else:
            # 硬截止（兼容 Level-1）
            if vov <= 0.0:
                return 0.0, 0.0, 0.0, 0.0
            vgsteff = vov
            s       = 1.0

        if vgsteff <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        # ── 2B. 迁移率衰减 kp_eff = kp/(1+θ·Vov)，仅对 Vov>0 生效 ─────────
        vov_pos = max(vov, 0.0)
        denom   = 1.0 + self.theta * vov_pos
        kp_eff  = self.kp / denom
        # dkp_eff/dVov (当 vov>0)
        dkp_dvov = -self.kp * self.theta / denom**2 if vov > 0.0 else 0.0

        # ── softmin(vds, vgsteff) ── Vdsat 平滑截断 ──────────────────────────
        vds_eff = self._softmin(vds, vgsteff)
        da      = self._softmin_da(vds, vgsteff)    # ∂Vds_eff/∂vds
        db      = self._softmin_db(vds, vgsteff)    # ∂Vds_eff/∂vgsteff

        # ── 漏极电流主体 ──────────────────────────────────────────────────────
        f   = vgsteff * vds_eff - 0.5 * vds_eff**2
        clm = 1.0 + self.lam * vds
        ID  = kp_eff * self.wl * f * clm

        # ── gm = ∂ID/∂VGS ─────────────────────────────────────────────────────
        # dVov/dVgs = 1, dVgsteff/dVgs = s
        # dVds_eff/dVgs = db·s
        # df/dVgs = s·Vds_eff + (Vgsteff-Vds_eff)·db·s = s·(Vds_eff + (Vgsteff-Vds_eff)·db)
        df_dvgs = s * (vds_eff + (vgsteff - vds_eff) * db)
        gm_val  = (dkp_dvov * self.wl * f + kp_eff * self.wl * df_dvgs) * clm

        # ── gds = ∂ID/∂VDS ────────────────────────────────────────────────────
        # dVov/dVds = +η (DIBL), dVgsteff/dVds = s·η
        # dVds_eff/dVds = da + db·s·η
        # df/dVds = s·η·Vds_eff + (Vgsteff-Vds_eff)·(da + db·s·η)
        eta_s     = self.eta * s
        dvds_eff  = da + db * eta_s
        df_dvds   = eta_s * vds_eff + (vgsteff - vds_eff) * dvds_eff
        gds_val   = ((dkp_dvov * self.eta * self.wl * f
                      + kp_eff * self.wl * df_dvds) * clm
                     + kp_eff * self.wl * f * self.lam)

        # ── gmb = ∂ID/∂VSB ────────────────────────────────────────────────────
        # dVov/dVsb = -dvth_body/dVsb = -dvth_dvsb
        # chain rule identical to gm but substituting dVov/dVsb
        dvov_dvsb = -dvth_dvsb
        df_dvsb   = dvov_dvsb * s * (vds_eff + (vgsteff - vds_eff) * db)
        gmb_val   = (dkp_dvov * dvov_dvsb * self.wl * f
                     + kp_eff * self.wl * df_dvsb) * clm

        return ID, gm_val, gds_val, gmb_val

    # ── BD/BS 结二极管 ────────────────────────────────────────────────────────

    def _eval_diodes(self, vbd, vbs):
        """
        BD 和 BS 结二极管的伴随模型参数（Newton-Raphson 线性化用）。

        参数
        ----
        vbd : V_B - V_D'  (衬底相对本征漏极电压，正偏时 >0)
        vbs : V_B - V_S'  (衬底相对本征源极电压，正偏时 >0)

        返回
        ----
        (IBD, GBD, IBD0, IBS, GBS, IBS0)

        IBD  : 二极管电流 IS·(exp(vbd/VT)-1)，方向 B→D'（正值流入 D'）
        GBD  : 增量电导 IS/VT·exp(vbd/VT)（恒正）
        IBD0 : 伴随电流源 = IBD - GBD·vbd
               （MNA 送值：A[D',D'] -= GBD；b[D'] -= IBD0）
        IBS, GBS, IBS0 : BS 结同理

        当 IS=0 时全部返回 0.0，不影响性能。
        """
        if self.IS == 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        VT = _VT_ROOM
        # 指数限幅：防止正偏时溢出（反偏时 vbd<0 不需要担心）
        vbd_c = min(float(vbd), 40.0 * VT)
        vbs_c = min(float(vbs), 40.0 * VT)

        exp_bd = np.exp(vbd_c / VT)
        exp_bs = np.exp(vbs_c / VT)

        IBD  = self.IS * (exp_bd - 1.0)
        GBD  = self.IS / VT * exp_bd
        IBD0 = IBD - GBD * vbd_c        # 伴随电流源

        IBS  = self.IS * (exp_bs - 1.0)
        GBS  = self.IS / VT * exp_bs
        IBS0 = IBS - GBS * vbs_c

        return IBD, GBD, IBD0, IBS, GBS, IBS0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def ids(self, vgs, vds, vsb=0.0):
        """漏极电流 ID (A)"""
        return self._eval(vgs, vds, vsb)[0]

    def gm(self, vgs, vds, vsb=0.0):
        """跨导 ∂ID/∂VGS (A/V)"""
        return self._eval(vgs, vds, vsb)[1]

    def gds(self, vgs, vds, vsb=0.0):
        """输出电导 ∂ID/∂VDS (A/V)"""
        return self._eval(vgs, vds, vsb)[2]

    def gmb(self, vgs, vds, vsb=0.0):
        """体效应跨导 ∂ID/∂VSB (A/V)"""
        return self._eval(vgs, vds, vsb)[3]

    def region(self, vgs, vds, vsb=0.0):
        """判断工作区域（定性，用于报告）"""
        vth_eff = self.vth - self.eta * vds
        if self.gamma > 0:
            vth_eff += self.gamma * (np.sqrt(abs(vsb) + self.phi_f)
                                     - np.sqrt(self.phi_f))
        vov = vgs - vth_eff
        if vov <= 0:
            return "Cutoff" if self.n_sub is None else "Subthreshold"
        if vds < vov - self.delta:
            return "Linear"
        if vds > vov + self.delta:
            return "Saturation"
        return "Transition"

    def capacitances(self, vgs, vds, vsb=0.0):
        """
        Compute small-signal NMOS gate capacitances (F) at operating point.

        Simplified charge-based partition (Meyer-like, smooth transitions):
          Cutoff    : Cgs = Cgso,          Cgd = Cgdo,          Cgb = Cgg
          Linear    : Cgs = Cgg/2 + Cgso,  Cgd = Cgg/2 + Cgdo,  Cgb = 0
          Saturation: Cgs = 2/3*Cgg + Cgso, Cgd = Cgdo,          Cgb = 0

        Smooth sigmoid blending avoids discontinuities between regions.

        Returns (Cgs, Cgd, Cgb) in Farads.
        Returns (0, 0, 0) if cgg is None (capacitance model disabled).
        """
        if self.cgg is None:
            return 0.0, 0.0, 0.0

        # Simplified Vov using Level-1 Vth (adequate for cap partition)
        vov = vgs - self.vth

        # Smooth step width (V) — a few kT/q so the transition is physically smooth
        w = 0.05

        # sig_on: 0 in cutoff, 1 in strong inversion
        sig_on = 1.0 / (1.0 + np.exp(-vov / w))

        # sig_sat: 0 in linear, 1 in saturation  (Vds crosses Vov)
        # Use max(vov,0) so sig_sat is well-defined for Vov<0 (sig_on~0 anyway)
        vdsat = max(float(vov), 1e-6)
        sig_sat = 1.0 / (1.0 + np.exp(-(vds - vdsat) / w))

        # Intrinsic channel cap partition
        cgs_int = self.cgg * sig_on * (0.5 + (2.0 / 3.0 - 0.5) * sig_sat)
        cgd_int = self.cgg * sig_on * 0.5 * (1.0 - sig_sat)
        cgb     = self.cgg * (1.0 - sig_on)

        return cgs_int + self.cgso, cgd_int + self.cgdo, cgb
