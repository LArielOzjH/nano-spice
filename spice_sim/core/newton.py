"""
Newton-Raphson 非线性求解器（含阻尼策略）。

算法（课程第11章）：
  x^(k+1) = x^k - λ_damp * J(x^k)⁻¹ * F(x^k)

其中：
  F(x)  = 非线性残差向量（由 MNABuilder.residual 提供）
  J(x)  = 雅可比矩阵（由 MNABuilder.build 的 A 矩阵近似，即线性化伴随模型）
  λ_damp = 阻尼因子，当残差增大时减半（确保收敛）
"""

import numpy as np


class NewtonSolver:
    """
    Newton-Raphson 迭代求解器。

    参数
    ----
    tol_v   : 节点电压收敛容差 (V)
    tol_i   : 电流收敛容差 (A)
    max_iter: 最大迭代次数
    verbose : 是否打印每步迭代信息
    """

    def __init__(self, tol_v=1e-9, tol_i=1e-12, max_iter=100, verbose=False):
        self.tol_v = tol_v
        self.tol_i = tol_i
        self.max_iter = max_iter
        self.verbose = verbose

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
        history  : 每步残差范数列表（用于报告）
        """
        x = x0.copy()
        history = []

        for k in range(self.max_iter):
            F = mna_builder.residual(x, vin, vdd)
            norm_F = np.linalg.norm(F)
            history.append(norm_F)

            if self.verbose:
                print(f"  iter {k:3d}: |F| = {norm_F:.3e}")

            # 收敛判断：电压残差和电流残差分别检查
            res_v = np.max(np.abs(F[[0, 1, 2]]))  # 节点电压方程残差
            res_i = np.max(np.abs(F[[3, 4]]))      # KVL 方程残差
            if res_v < self.tol_i and res_i < self.tol_v:
                return x, k + 1, True, history

            # 构建雅可比（线性化 MNA）
            A, b = mna_builder.build(x, vin, vdd)

            # 求解线性方程组：A * Δx = -F
            try:
                dx = np.linalg.solve(A, -F)
            except np.linalg.LinAlgError:
                # 矩阵奇异，收敛失败
                return x, k + 1, False, history

            # 阻尼策略：若步长导致残差增大，减半阻尼因子
            lam = 1.0
            for _ in range(10):
                x_new = x + lam * dx
                F_new = mna_builder.residual(x_new, vin, vdd)
                if np.linalg.norm(F_new) <= norm_F + 1e-12:
                    break
                lam *= 0.5

            x = x + lam * dx

        return x, self.max_iter, False, history
