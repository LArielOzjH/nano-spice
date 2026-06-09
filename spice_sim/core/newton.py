"""
Newton-Raphson 非线性求解器（含阻尼策略 + SPICE 标准收敛判据）。

算法（课程第11章）：
  x^(k+1) = x^k + λ_damp · (-J(x^k)⁻¹ · F(x^k))

收敛判据（SPICE 标准，第11章 p.12）：
  对每个分量 i：
    |x^(k+1)[i] - x^k[i]| < ε_a[i] + ε_r · min{|x^k[i]|, |x^(k+1)[i]|}

  其中：
    ε_a = 1×10⁻⁶ V（电压变量）/ 1×10⁻¹² A（电流变量）
    ε_r = 0.001（相对容差，与 SPICE 程序一致）

  所有分量同时满足则判定收敛。
"""

import numpy as np

# SPICE 标准容差
_EPS_A_V = 1e-6   # 电压绝对容差 (V)
_EPS_A_I = 1e-12  # 电流绝对容差 (A)
_EPS_R   = 1e-3   # 相对容差（0.1%）

# 7 变量中：前 5 个是电压，后 2 个是电流
_EPS_A = np.array([_EPS_A_V] * 5 + [_EPS_A_I] * 2)


class NewtonSolver:
    """
    Newton-Raphson 迭代求解器。

    参数
    ----
    eps_r   : 相对容差（默认 0.001，即 SPICE 标准值）
    max_iter: 最大迭代次数
    verbose : 是否打印每步迭代信息
    """

    def __init__(self, eps_r=_EPS_R, max_iter=100, verbose=False):
        self.eps_r    = eps_r
        self.max_iter = max_iter
        self.verbose  = verbose

    def solve(self, mna_builder, x0, vin, vdd):
        """
        求解 F(x) = 0。

        参数
        ----
        mna_builder : MNABuilder 实例
        x0          : 初始猜测向量
        vin, vdd    : 电路参数（已知电压）

        返回
        ----
        x        : 收敛解向量
        n_iter   : 实际迭代次数
        converged: 是否收敛
        history  : 每步残差范数列表（用于调试/报告）
        """
        x = x0.copy()
        history = []

        for k in range(self.max_iter):
            F      = mna_builder.residual(x, vin, vdd)
            norm_F = np.linalg.norm(F)
            history.append(norm_F)

            if self.verbose:
                print(f"  iter {k:3d}: |F| = {norm_F:.3e}")

            # 构建线性化雅可比（伴随模型）
            A, b = mna_builder.build(x, vin, vdd)

            # 求解线性步：J·Δx = -F
            try:
                dx = np.linalg.solve(A, -F)
            except np.linalg.LinAlgError:
                return x, k + 1, False, history

            # 阻尼策略：若步长导致残差增大，则将步长减半（最多 10 次）
            lam = 1.0
            for _ in range(10):
                x_new  = x + lam * dx
                norm_new = np.linalg.norm(mna_builder.residual(x_new, vin, vdd))
                if norm_new <= norm_F + 1e-12:
                    break
                lam *= 0.5

            x_new = x + lam * dx

            # ── SPICE 标准收敛判据 ────────────────────────────────────────
            # |Δx[i]| < ε_a[i] + ε_r · min(|x[i]|, |x_new[i]|)
            delta     = np.abs(x_new - x)
            threshold = _EPS_A + self.eps_r * np.minimum(np.abs(x), np.abs(x_new))

            x = x_new

            if np.all(delta < threshold):
                return x, k + 1, True, history

        return x, self.max_iter, False, history
