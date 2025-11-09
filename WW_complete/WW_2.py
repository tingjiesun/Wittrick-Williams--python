"""
Wittrick-Williams算法的高级Python实现示例
包含完整的动力刚度矩阵计算和更详细的测试用例

这个示例展示了如何在实际工程中应用Wittrick-Williams算法
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class BeamElement:
    """梁单元类型定义（更完整版本）"""
    element_id: int
    node_i: int  # i结点编号
    node_j: int  # j结点编号
    length: float  # 单元长度
    EA: float  # 轴向刚度
    EI: float  # 弯曲刚度
    mass_per_length: float  # 单位长度质量
    cos_alpha: float = 0.0  # 方向余弦
    sin_alpha: float = 0.0  # 方向正弦

    def __post_init__(self):
        """计算派生属性"""
        self.mass_total = self.mass_per_length * self.length


class DynamicStiffnessMatrix:
    """动力刚度矩阵计算类"""

    @staticmethod
    def element_dynamic_stiffness(element: BeamElement, omega: float) -> np.ndarray:
        """
        计算单个梁单元的动力刚度矩阵

        Args:
            element: 梁单元
            omega: 圆频率 (rad/s)

        Returns:
            6x6动力刚度矩阵
        """
        L = element.length
        EA = element.EA
        EI = element.EI
        m = element.mass_per_length

        # 无量纲频率参数
        omega_squared = omega ** 2
        alpha_squared = omega_squared * m * L ** 4 / EI  # 弯曲频率参数
        beta_squared = omega_squared * m * L ** 2 / EA  # 轴向频率参数

        # 动力刚度矩阵系数
        if alpha_squared > 1e-10:  # 避免数值问题
            alpha = math.sqrt(alpha_squared)
            s = math.sin(alpha)
            c = math.cos(alpha)
            sh = math.sinh(alpha)
            ch = math.cosh(alpha)

            # 弯曲动力刚度系数
            denom = 2 * (1 - c * ch)
            if abs(denom) < 1e-12:
                # 接近共振频率时的处理
                k11 = k22 = EI / L ** 3 * 12
                k12 = k21 = EI / L ** 2 * 6
            else:
                k11 = k22 = EI / L ** 3 * alpha ** 4 * (s * sh) / denom
                k12 = k21 = EI / L ** 2 * alpha ** 2 * (s * ch - c * sh) / denom
        else:
            # 静力刚度（低频近似）
            k11 = k22 = 12 * EI / L ** 3
            k12 = k21 = 6 * EI / L ** 2

        # 轴向动力刚度
        if beta_squared > 1e-10:
            beta = math.sqrt(beta_squared)
            k_axial = EA / L * beta / math.tan(beta)
        else:
            k_axial = EA / L

        # 组装6x6动力刚度矩阵（局部坐标系）
        K_local = np.zeros((6, 6))

        # 轴向刚度
        K_local[0, 0] = K_local[3, 3] = k_axial
        K_local[0, 3] = K_local[3, 0] = -k_axial

        # 弯曲刚度
        K_local[1, 1] = K_local[4, 4] = k11
        K_local[1, 4] = K_local[4, 1] = -k11
        K_local[2, 2] = K_local[5, 5] = k22
        K_local[2, 5] = K_local[5, 2] = k12
        K_local[1, 2] = K_local[2, 1] = k12
        K_local[4, 5] = K_local[5, 4] = k21
        K_local[1, 5] = K_local[5, 1] = -k12
        K_local[2, 4] = K_local[4, 2] = -k21

        return K_local

    @staticmethod
    def transformation_matrix(cos_alpha: float, sin_alpha: float) -> np.ndarray:
        """
        计算坐标转换矩阵

        Args:
            cos_alpha: 方向余弦
            sin_alpha: 方向正弦

        Returns:
            6x6转换矩阵
        """
        T = np.zeros((6, 6))

        # 2x2转换子矩阵
        T_sub = np.array([[cos_alpha, sin_alpha],
                          [-sin_alpha, cos_alpha]])

        # 填充6x6矩阵
        T[0:2, 0:2] = T_sub
        T[2, 2] = 1.0
        T[3:5, 3:5] = T_sub
        T[5, 5] = 1.0

        return T


class AdvancedWittrickWilliams:
    """高级Wittrick-Williams算法实现"""

    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
        self.pi = math.pi
        self.dynamic_stiffness = DynamicStiffnessMatrix()

    def calculate_j0(self, freq: float, elements: List[BeamElement]) -> int:
        """计算J0值(改进版本)"""
        omega = 2 * self.pi * freq  # 转换为圆频率
        j0 = 0

        for elem in elements:
            L = elem.length
            EA = elem.EA
            EI = elem.EI
            m = elem.mass_per_length

            # 轴向振动特征值个数
            nu = omega * L * math.sqrt(m / EA)
            ja = int(nu / self.pi)

            # 弯曲振动特征值个数
            lambda_val = L * (omega ** 2 * m / EI) ** 0.25

            if lambda_val > 1e-6:
                inv_e = math.exp(-lambda_val)
                cos_lambda = math.cos(lambda_val)

                # 改进的符号判断
                discriminant = inv_e - cos_lambda * (1.0 + inv_e ** 2) / 2.0
                sg = 1 if discriminant > 0 else -1

                n_lambda = int(lambda_val / self.pi)
                jb = n_lambda - (1 - (-1) ** n_lambda * sg) // 2
            else:
                jb = 0

            j0 += ja + jb

        return max(0, j0)  # 确保非负

    def assemble_global_dynamic_stiffness(self, elements: List[BeamElement],
                                          omega: float, num_dofs: int) -> np.ndarray:
        """
        组装全局动力刚度矩阵

        Args:
            elements: 单元列表
            omega: 圆频率
            num_dofs: 总自由度数

        Returns:
            全局动力刚度矩阵
        """
        K_global = np.zeros((num_dofs, num_dofs))

        for elem in elements:
            # 计算单元动力刚度矩阵
            K_local = self.dynamic_stiffness.element_dynamic_stiffness(elem, omega)

            # 坐标转换
            T = self.dynamic_stiffness.transformation_matrix(elem.cos_alpha, elem.sin_alpha)
            K_global_elem = T.T @ K_local @ T

            # 组装到全局矩阵（简化版本，假设连续编号）
            dof_indices = [
                3 * (elem.node_i - 1), 3 * (elem.node_i - 1) + 1, 3 * (elem.node_i - 1) + 2,
                3 * (elem.node_j - 1), 3 * (elem.node_j - 1) + 1, 3 * (elem.node_j - 1) + 2
            ]

            for i in range(6):
                for j in range(6):
                    if dof_indices[i] < num_dofs and dof_indices[j] < num_dofs:
                        K_global[dof_indices[i], dof_indices[j]] += K_global_elem[i, j]

        return K_global

    def calculate_jk(self, freq: float, elements: List[BeamElement],
                     num_dofs: int) -> int:
        """计算JK值(完整版本)"""
        omega = 2 * self.pi * freq

        try:
            # 组装全局动力刚度矩阵
            K_global = self.assemble_global_dynamic_stiffness(elements, omega, num_dofs)

            # 计算特征值
            eigenvalues = np.linalg.eigvals(K_global)

            # 统计负特征值个数
            jk = np.sum(np.real(eigenvalues) < -1e-10)

            return int(jk)

        except Exception as e:
            print(f"JK计算出现错误: {e}")
            return 0

    def find_frequency_bounds(self, k_order: int, elements: List[BeamElement],
                              num_dofs: int) -> Tuple[float, float]:
        """寻找频率搜索边界"""
        freq_lower = 0.0
        freq_upper = 10.0

        # 寻找下界
        max_iterations = 50
        for _ in range(max_iterations):
            j0 = self.calculate_j0(freq_lower, elements)
            jk = self.calculate_jk(freq_lower, elements, num_dofs)
            total_j = j0 + jk

            if total_j < k_order:
                break
            freq_lower /= 2.0

            if freq_lower < 1e-6:
                break

        # 寻找上界
        for _ in range(max_iterations):
            j0 = self.calculate_j0(freq_upper, elements)
            jk = self.calculate_jk(freq_upper, elements, num_dofs)
            total_j = j0 + jk

            if total_j >= k_order:
                break
            freq_upper *= 2.0

            if freq_upper > 1e6:
                break

        return freq_lower, freq_upper

    def calculate_k_freq(self, k_order: int, elements: List[BeamElement],
                         num_dofs: int) -> float:
        """计算第k阶频率(改进版本)"""
        freq_lower, freq_upper = self.find_frequency_bounds(k_order, elements, num_dofs)

        max_iterations = 1000
        for iteration in range(max_iterations):
            freq_mid = (freq_lower + freq_upper) / 2.0

            j0 = self.calculate_j0(freq_mid, elements)
            jk = self.calculate_jk(freq_mid, elements, num_dofs)
            total_j = j0 + jk

            if total_j >= k_order:
                freq_upper = freq_mid
            else:
                freq_lower = freq_mid

            # 收敛判断
            relative_error = (freq_upper - freq_lower) / (1.0 + freq_upper)
            if relative_error <= self.tolerance:
                break

        return (freq_lower + freq_upper) / 2.0

# ------------------------------------------------------------------------------------------------
elements= [
        BeamElement(element_id=1, node_i=1, node_j=2, length=1.0,
                    EA=1.0, EI=1.0, mass_per_length=1,
                    cos_alpha=1.0, sin_alpha=0.0),
    ]
aw = AdvancedWittrickWilliams()

# 在 AdvancedWittrickWilliams.calculate_j0 内部添加：
# if EI <= 0: jb = 0
j0 = aw.calculate_j0(freq=0.125, elements=elements)
print("j0 =", j0)  # 结果：0

#-----------------------------------------------------------------------------------------------
def create_cantilever_beam_example() -> Tuple[List[BeamElement], int]:
    """
    创建悬臂梁示例

    Returns:
        elements: 单元列表
        num_dofs: 自由度总数
    """
    # 材料和几何参数
    E = 1.0  # 弹性模量 (Pa)
    rho = 1.0  # 密度 (kg/m³)
    L_total = 1.0  # 总长度 (m)
    b = 0.02  # 宽度 (m)
    h = 0.01  # 高度 (m)

    A = b * h  # 截面积
    I = 1.0  # 惯性矩
    EA = 1.0
    EI =1.0
    mass_per_length = 1.0

    # 创建单元（将梁分为4个单元）
    num_elements = 4
    element_length = L_total / num_elements

    elements = []
    for i in range(num_elements):
        element = BeamElement(
            element_id=i + 1,
            node_i=i + 1,
            node_j=i + 2,
            length=element_length,
            EA=EA,
            EI=EI,
            mass_per_length=mass_per_length,
            cos_alpha=1.0,  # 水平梁
            sin_alpha=0.0
        )
        elements.append(element)

    # 自由度数（5个节点，每个节点3个自由度，但第1个节点固定）
    num_dofs = 3 * (num_elements + 1) - 3  # 减去固定端的3个自由度

    return elements, num_dofs


def run_frequency_analysis():
    """运行频率分析示例"""
    print("=== 高级Wittrick-Williams算法频率分析 ===\n")

    # 创建悬臂梁模型
    elements, num_dofs = create_cantilever_beam_example()

    print(f"模型信息:")
    print(f"  单元数量: {len(elements)}")
    print(f"  自由度数: {num_dofs}")
    print(f"  总长度: {sum(elem.length for elem in elements):.2f} m")
    print(f"  EI: {elements[0].EI:.2e} N⋅m²")
    print(f"  质量/长度: {elements[0].mass_per_length:.2f} kg/m\n")

    # 创建算法实例
    ww = AdvancedWittrickWilliams(tolerance=1e-6)

    # 计算前几阶频率
    frequencies = []
    num_modes = 5

    print("正在计算自然频率...")
    for k in range(1, num_modes + 1):
        try:
            freq = ww.calculate_k_freq(k, elements, num_dofs)
            frequencies.append(freq)
            print(f"第{k}阶频率: {freq:.4f} Hz")
        except Exception as e:
            print(f"第{k}阶频率计算失败: {e}")
            break

    # 理论解对比（悬臂梁的理论频率）
    print(f"\n理论解对比（悬臂梁）:")
    L = sum(elem.length for elem in elements)
    EI = elements[0].EI
    m = elements[0].mass_per_length

    # 悬臂梁的特征值
    lambda_values = [1.875, 4.694, 7.855, 10.996, 14.137]

    for i, lam in enumerate(lambda_values[:len(frequencies)]):
        if i < len(frequencies):
            freq_theory = (lam ** 2) / (2 * math.pi * L ** 2) * math.sqrt(EI / m)
            error = abs(frequencies[i] - freq_theory) / freq_theory * 100
            print(f"第{i + 1}阶: 理论值 {freq_theory:.4f} Hz, "
                  f"计算值 {frequencies[i]:.4f} Hz, 误差 {error:.2f}%")

    return frequencies


def plot_frequency_convergence():
    """绘制频率收敛性分析"""
    try:
        import matplotlib.pyplot as plt

        # 不同单元数的分析
        element_counts = [2, 4, 8, 16]
        first_frequencies = []

        print(f"\n=== 收敛性分析 ===")

        for n_elem in element_counts:
            # 创建不同单元数的模型
            E = 2.1e11
            rho = 7850
            L_total = 1.0
            b, h = 0.02, 0.01
            A, I = b * h, b * h ** 3 / 12

            elements = []
            element_length = L_total / n_elem

            for i in range(n_elem):
                element = BeamElement(
                    element_id=i + 1,
                    node_i=i + 1,
                    node_j=i + 2,
                    length=element_length,
                    EA=E * A,
                    EI=E * I,
                    mass_per_length=rho * A,
                    cos_alpha=1.0,
                    sin_alpha=0.0
                )
                elements.append(element)

            num_dofs = 3 * (n_elem + 1) - 3
            ww = AdvancedWittrickWilliams(tolerance=1e-6)

            try:
                freq1 = ww.calculate_k_freq(1, elements, num_dofs)
                first_frequencies.append(freq1)
                print(f"单元数 {n_elem:2d}: 第1阶频率 = {freq1:.4f} Hz")
            except:
                first_frequencies.append(None)
                print(f"单元数 {n_elem:2d}: 计算失败")

        # 绘制收敛图
        valid_counts = []
        valid_freqs = []
        for i, freq in enumerate(first_frequencies):
            if freq is not None:
                valid_counts.append(element_counts[i])
                valid_freqs.append(freq)

        if len(valid_freqs) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(valid_counts, valid_freqs, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('单元数量')
            plt.ylabel('第1阶频率 (Hz)')
            plt.title('Wittrick-Williams算法收敛性分析')
            plt.grid(True, alpha=0.3)
            plt.show()

    except ImportError:
        print("matplotlib未安装，跳过绘图")


if __name__ == "__main__":
    # 运行主要分析
    frequencies = run_frequency_analysis()

    # 运行收敛性分析
    plot_frequency_convergence()

    print(f"\n=== 算法特点总结 ===")
    print("1. Wittrick-Williams算法是一种精确的频率计算方法")
    print("2. 通过计算特征值个数来确定频率，避免了遗漏模态")
    print("3. 使用二分法确保收敛到正确的频率值")
    print("4. 适用于复杂结构的动力学分析")
    print("5. 本实现包含了完整的动力刚度矩阵计算")
