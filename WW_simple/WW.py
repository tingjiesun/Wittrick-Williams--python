"""
Wittrick-Williams算法的Python实现
用于结构动力学中自然频率的计算

该算法基于Wittrick-Williams定理,通过计算特征值个数来确定结构的自然频率。
主要包含三个核心函数：
1. calculate_j0: 计算频率下界的特征值个数
2. calculate_jk: 计算动力刚度矩阵的负特征值个数
3. calculate_k_freq: 使用二分法求解第k阶自然频率
"""

import numpy as np
import math
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Element:
    """单元类型定义"""
    length: float  # 单元长度
    EA: float  # 轴向刚度 (弹性模量×截面积)
    EI: float  # 弯曲刚度 (弹性模量×惯性矩)
    mass: float  # 单位长度质量
    joint_no: List[int] = None  # 结点编号
    cos_a: float = 0.0  # 方向余弦
    sin_a: float = 0.0  # 方向正弦
    glb_dof: List[int] = None  # 全局自由度编号


class WittrickWilliams:
    """Wittrick-Williams算法实现类"""

    def __init__(self, tolerance: float = 1e-6):
        """
        初始化Wittrick-Williams算法
        Args:   #标题，表示下面列出的是函数的输入参数
            tolerance: 收敛容差    # 用于控制数值计算的精度要求,默认值为1e-6
        """
        self.tolerance = tolerance
        self.pi = math.pi

# -----------------------------------------------------------------------------------------------
    def calculate_j0(self, freq: float, elements: List[Element]) -> int:
        """
        计算J0值 - 频率下界特征值个数
        使用Wittrick-Williams算法计算频率下界的特征值个数

        Args:
            freq: 频率    # 待计算的自然频率值
            elements: 单元数组    # 结构中所有单元的列表

        Returns:    # 标题，表示下面列出的是函数的返回值
            J0: 频率下界特征值个数    # 计算得到的频率下界的特征值个数
        """
        j0 = 0

        # 遍历所有单元，累计各单元的特征值个数
        for elem in elements:
            # 计算无量纲频率参数（ 轴向振动的无量纲频率nu 和 弯曲振动的无量纲频率λ ）
            nu = freq * elem.length * math.sqrt(elem.mass / elem.EA)  # 轴向振动的无量纲频率nu=频率×单元长度×√（m除以EA）
            lambda_val = elem.length * (
                        (freq ** 2 * elem.mass / elem.EI) ** 0.25)  # 弯曲振动的无量纲频率λ=单元长度×（频率的平方×m除以EI）的0.25次方  书p185页

            # 计算轴向振动的特征值个数
            ja = int(nu / math.acos(-1.0))  # acos(-1) = π

            # 计算弯曲振动的特征值个数
            # inv_e = math.exp(-lambda_val)
            # 等价下式 sg = int(math.copysign(1.0, inv_e - math.cos(lambda_val) * (1.0 + inv_e ** 2) / 2.0))
            sg = int(math.copysign(1.0, 1 - math.cosh(lambda_val) * math.cos(lambda_val)))  # 书p194
            jb = int(lambda_val / self.pi) - (1 - (-1) ** int(lambda_val / self.pi) * sg) // 2

            j0 += ja + jb

        return j0  # 返回频率下界的特征值个数（固端频率数）

# -----------------------------------------------------------------------------------------------
    def calculate_jk(self, freq: float, elements: List[Element],
                     stiffness_matrix_func=None) -> int:
        """
        计算JK值 - 动力刚度矩阵的负特征值个数（频率数）

        Args:
            freq: 频率
            elements: 单元数组
            stiffness_matrix_func: 动力刚度矩阵计算函数

        Returns:
            Jk: 负特征值个数
        """
        # 这里需要实现动力刚度矩阵的组装和分解
        # 由于完整的矩阵操作较为复杂，这里提供简化版本

        if stiffness_matrix_func is None:
            # 简化实现：假设已有动力刚度矩阵的特征值
            # 实际应用中需要组装完整的动力刚度矩阵
            return self._simplified_jk_calculation(freq, elements)
        else:
            # 使用提供的刚度矩阵函数
            K_dynamic = stiffness_matrix_func(freq, elements)
            eigenvalues = np.linalg.eigvals(K_dynamic)
            return np.sum(eigenvalues < 0)

    def _simplified_jk_calculation(self, freq: float, elements: List[Element]) -> int:
        """
        简化的JK计算(用于演示)
        实际应用中需要完整的矩阵组装和分解过程
        """
        # 这是一个简化版本，实际需要完整的动力刚度矩阵计算
        jk = 0
        for elem in elements:
            # 简化的负特征值估算
            omega_squared = freq ** 2
            critical_freq_squared = elem.EI / (elem.mass * elem.length ** 4)
            if omega_squared > critical_freq_squared:
                jk += 1
        return jk

    def calculate_k_freq(self, k_freq_order: int, elements: List[Element],
                         stiffness_matrix_func=None) -> float:
        """
        计算第k阶频率
        使用二分法求解第k阶自然频率

        Args:
            k_freq_order: 频率阶数
            elements: 单元数组
            stiffness_matrix_func: 动力刚度矩阵计算函数

        Returns:
            freq: 第k阶频率
        """
        # 确定频率搜索的下界
        freq1 = 1.0
        freq2 = 10.0

        # 寻找下界
        while True:
            j0 = self.calculate_j0(freq1, elements)
            jk = self.calculate_jk(freq1, elements, stiffness_matrix_func)
            total_j = j0 + jk
            if total_j < k_freq_order:
                break
            freq1 /= 2.0

        # 寻找上界
        while True:
            j0 = self.calculate_j0(freq2, elements)
            jk = self.calculate_jk(freq2, elements, stiffness_matrix_func)
            total_j = j0 + jk
            if total_j > k_freq_order:
                break
            freq2 *= 2.0

        # 使用二分法精确求解频率
        while True:
            freq = (freq1 + freq2) / 2.0
            j0 = self.calculate_j0(freq, elements)
            jk = self.calculate_jk(freq, elements, stiffness_matrix_func)
            total_j = j0 + jk

            if total_j >= k_freq_order:
                freq2 = freq
            else:
                freq1 = freq

            if (freq2 - freq1) <= self.tolerance * (1.0 + freq2):
                break

        return (freq1 + freq2) / 2.0

    def calculate_frequencies(self, freq_start: int, num_freqs: int,
                              elements: List[Element],
                              stiffness_matrix_func=None) -> List[float]:
        """
        计算多阶频率

        Args:
            freq_start: 起始阶数
            num_freqs: 频率个数
            elements: 单元数组
            stiffness_matrix_func: 动力刚度矩阵计算函数

        Returns:
            frequencies: 频率数组
        """
        frequencies = []

        # 逐个计算各阶频率
        for k in range(freq_start, freq_start + num_freqs):
            freq = self.calculate_k_freq(k, elements, stiffness_matrix_func)
            frequencies.append(freq)

        return frequencies


# -----------------------------------------------------------------------------------------------
def create_sample_element(length: float = 1.0, EA: float = 1e6,
                          EI: float = 1e4, mass: float = 100.0) -> Element:
    """
    创建示例单元

    Args:
        length: 单元长度
        EA: 轴向刚度
        EI: 弯曲刚度
        mass: 单位长度质量

    Returns:
        Element: 单元对象
    """
    return Element(
        length=length,
        EA=EA,
        EI=EI,
        mass=mass,
        joint_no=[1, 2],
        cos_a=1.0,
        sin_a=0.0,
        glb_dof=[1, 2, 3, 4, 5, 6]  # 全局自由度编号
    )


# 示例使用
if __name__ == "__main__":
    # 创建Wittrick-Williams算法实例
    ww = WittrickWilliams(tolerance=1e-6)  # 继承类

    # 创建示例单元
    elements = [
        create_sample_element(length=2.0, EA=2e6, EI=1e4, mass=150.0),
        create_sample_element(length=1.5, EA=1.5e6, EI=8e3, mass=120.0)
    ]

    # 测试频率计算
    test_freq = 10.0

    print("=== Wittrick-Williams算法测试 ===")
    print(f"测试频率: {test_freq} Hz")
    print(f"单元数量: {len(elements)}")

    # 计算J0值
    j0 = ww.calculate_j0(test_freq, elements)
    print(f"J0 (频率下界特征值个数): {j0}")

    # 计算JK值（简化版本）
    jk = ww.calculate_jk(test_freq, elements)
    print(f"JK (负特征值个数): {jk}")

    # 计算第1阶频率
    try:
        first_freq = ww.calculate_k_freq(1, elements)
        print(f"第1阶自然频率: {first_freq:.4f} Hz")
    except Exception as e:
        print(f"频率计算出现问题: {e}")

    # 计算前3阶频率
    try:
        frequencies = ww.calculate_frequencies(1, 3, elements)
        print("前3阶自然频率:")
        for i, freq in enumerate(frequencies, 1):
            print(f"  第{i}阶: {freq:.4f} Hz")
    except Exception as e:
        print(f"多阶频率计算出现问题: {e}")

    print("\n注意：这是Wittrick-Williams算法的基础实现")
    print("实际工程应用中需要完整的动力刚度矩阵计算和矩阵分解功能")
