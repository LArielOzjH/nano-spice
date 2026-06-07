"""
线性电阻器件模型及 MNA stamp。
"""


class Resistor:
    """
    线性电阻。

    参数
    ----
    resistance : 阻值 (Ω)
    node_p     : 正端节点编号（0 = GND）
    node_n     : 负端节点编号
    """

    def __init__(self, resistance, node_p, node_n):
        self.R = float(resistance)
        self.G = 1.0 / self.R
        self.node_p = node_p
        self.node_n = node_n

    def stamp(self, G_mat):
        """将电导值注入 MNA 的 G 子矩阵（去除 GND 行列后的矩阵）。"""
        p, n = self.node_p, self.node_n
        if p != 0:
            G_mat[p - 1, p - 1] += self.G
        if n != 0:
            G_mat[n - 1, n - 1] += self.G
        if p != 0 and n != 0:
            G_mat[p - 1, n - 1] -= self.G
            G_mat[n - 1, p - 1] -= self.G
