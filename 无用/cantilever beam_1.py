#书P197————轴向振动悬臂梁的频率计算
#注意：仅轴向振动，J0=Ja（Jb省略）

import math
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Element:
    """
    梁单元数据结构（均匀直梁，欧拉-伯努利假设）。

    字段:
    - Len: 单元长度 L
    - EA: 轴向刚度 EA
    - EI: 弯曲刚度 EI
    - mass: 线密度 m（单位长度质量）
    - CosA, SinA: 单元方向余弦与正弦（局部→全局坐标）
    - GlbDOF: 6 个全局自由度编号（1-based；约束用 0）
    """
    Len: float
    EA: float
    EI: float
    mass: float
    CosA: float
    SinA: float
    GlbDOF: List[int]


def _trans_matrix(cos_a: float, sin_a: float) -> np.ndarray:
    """
    构造 6×6 的坐标变换矩阵 ET（局部 → 全局）。

    每节点 3 自由度 [u, v, θ]，位移部分按平面旋转 R；
    角度 θ 保持不变（小变形假设）。
    返回的矩阵为两块 3×3 的 R 的块对角结构。
    """
    R = np.array([[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    ET = np.zeros((6, 6))
    ET[0:3, 0:3] = R
    ET[3:6, 3:6] = R
    return ET


def _ed_stiffness_matrix(elem: Element, freq: float) -> np.ndarray:
    """
    计算单元在给定圆频率下的动力刚度矩阵 EK（6×6）。

    模型:
    - 轴向 + 弯曲的频率相关刚度，欧拉-伯努利梁假设，线性小振幅。

    自由度次序:
    - [ui, vi, θi, uj, vj, θj]

    参数:
    - elem: 单元参数（EA, EI, m, L, 方向余弦, GlbDOF）
    - freq: 圆频率 ω（rad/s）

    返回:
    - 6×6 的局部坐标 EK
    """
    EAL = elem.EA / elem.Len
    EIL = elem.EI / elem.Len
    EIL2 = elem.EI / (elem.Len ** 2)
    EIL3 = elem.EI / (elem.Len ** 3)
    nu = freq * elem.Len * math.sqrt(elem.mass / elem.EA)  # 轴向无量纲频率
    lam = elem.Len * ((freq ** 2 * elem.mass / elem.EI) ** 0.25)  # 弯曲无量纲频率
    sl = math.sin(lam)
    cl = math.cos(lam)
    inve = math.exp(-lam)  # e^{-λ}
    esh = (1.0 - inve ** 2) / 2.0  # 与 sinh(λ) 相关的组合项
    ech = (1.0 + inve ** 2) / 2.0  # 与 cosh(λ) 相关的组合项
    phi = inve - ech * cl  # 组合项，接近 0 时数值可能不稳定
    B1 = nu / math.tan(nu)
    B2 = nu / math.sin(nu)
    T = (lam ** 3) * (sl * ech + cl * esh) / phi
    R = (lam ** 3) * (esh + inve * sl) / phi
    Q = (lam ** 2) * (esh * sl) / phi
    H = (lam ** 2) * (ech - inve * cl) / phi
    S = lam * (sl * ech - cl * esh) / phi
    C = lam * (esh - inve * sl) / phi
    EK = np.zeros((6, 6))
    EK[0] = np.array([B1 * EAL, 0.0, 0.0, -B2 * EAL, 0.0, 0.0])
    EK[1] = np.array([0.0, T * EIL3, Q * EIL2, 0.0, -R * EIL3, H * EIL2])
    EK[2] = np.array([0.0, Q * EIL2, S * EIL, 0.0, -H * EIL2, C * EIL])
    EK[3] = np.array([-B2 * EAL, 0.0, 0.0, B1 * EAL, 0.0, 0.0])
    EK[4] = np.array([0.0, -R * EIL3, -H * EIL2, 0.0, T * EIL3, -Q * EIL2])
    EK[5] = np.array([0.0, H * EIL2, C * EIL, 0.0, -Q * EIL2, S * EIL])
    return EK


def _gd_stiffness_matrix(elements: List[Element], freq: float, n_glb_dof: int) -> np.ndarray:
    """
    组装给定频率下的全局动力刚度矩阵 K（n_glb_dof × n_glb_dof）。

    步骤:
    - 逐单元计算局部 EK
    - 用 ET 做坐标变换得到 EKg
    - 按 GlbDOF 将 EKg 汇入全局矩阵 K

    约定:
    - GlbDOF 使用 1-based；值为 0 的自由度表示该自由度被约束

    注意:
    - 当前实现同时填充 K[i,j] 与 K[j,i] 以保持矩阵对称；
      在 i、j 双索引遍历下，非对角项会被累加两次。若需避免重复，
      可仅填充一次或限制遍历 i ≤ j 并最终对称化。
    """
    K = np.zeros((n_glb_dof, n_glb_dof))
    for elem in elements:
        EK = _ed_stiffness_matrix(elem, freq)
        ET = _trans_matrix(elem.CosA, elem.SinA)
        EKg = ET.T @ EK @ ET
        ev = elem.GlbDOF
        for j in range(6):
            JG = ev[j]
            if JG <= 0:
                continue
            jidx = JG - 1
            for i in range(6):
                IG = ev[i]
                if IG <= 0:
                    continue
                iidx = IG - 1
                val = EKg[i, j]
                K[iidx, jidx] += val
                K[jidx, iidx] += val
    return K


def calculate_j0(freq: float, elements: List[Element]) -> int:
    """
    计算 J0: 频率下界的特征值个数（Wittrick–Williams 定理）。

    原理:
    - 对轴向: ja = ⌊ν/π⌋，其中 ν = ω L √(m/EA)
    - 对弯曲: jb 由 λ = L(ω² m/EI)^{1/4} 的区间计数与符号项组合得到
      采用 inve = e^{-λ} 与 cos(λ) 的判定构造 sg ∈ {+1, −1}

    返回:
    - 各单元 ja+jb 的和，即结构在该频率的下界特征值计数
    """
    pi = math.acos(-1.0)
    j0 = 0
    for elem in elements:
        nu = freq * elem.Len * math.sqrt(elem.mass / elem.EA)
        lam = elem.Len * ((freq ** 2 * elem.mass / elem.EI) ** 0.25)
        ja = int(nu / pi)
        inve = math.exp(-lam)
        sg = 1 if (inve - math.cos(lam) * (1.0 + inve ** 2) / 2.0) >= 0.0 else -1
        n_int = int(lam / pi)
        parity = (-1) ** n_int
        jb = n_int - (1 - parity * sg) // 2
        j0 += ja
    return j0


def calculate_jk(freq: float, elements: List[Element], n_glb_dof: int) -> int:
    """
    计算 JK: 全局动力刚度矩阵 K 的负特征值个数

    """
    K = _gd_stiffness_matrix(elements, freq, n_glb_dof)
    K = (K + K.T) * 0.5
    w = np.linalg.eigvalsh(K)
    return int(np.sum(w < -1e-12))


def calculate_kfreq(kfreq: int, toler: float, elements: List[Element], n_glb_dof: int) -> float:
    """
    用二分法求结构的第 k 阶圆频率（rad/s）
    参数:
    - kfreq: 目标阶次 k
    - toler: 收敛阈值
    - elements, n_glb_dof: 结构模型

    返回:
    - 第 k 阶圆频率（rad/s）
    """
    freq1 = 1.0
    freq2 = 10.0
    while True:
        j0 = calculate_j0(freq1, elements)
        jk = calculate_jk(freq1, elements, n_glb_dof)
        if j0 + jk < kfreq:
            break
        freq1 *= 0.5
    while True:
        j0 = calculate_j0(freq2, elements)
        jk = calculate_jk(freq2, elements, n_glb_dof)
        if j0 + jk > kfreq:
            break
        freq2 *= 2.0
    while True:
        freq = 0.5 * (freq1 + freq2)
        j0 = calculate_j0(freq, elements)
        jk = calculate_jk(freq, elements, n_glb_dof)
        if j0 + jk >= kfreq:
            freq2 = freq
        else:
            freq1 = freq
        if (freq2 - freq1) <= toler * (1.0 + freq2):
            break
    return 0.5 * (freq1 + freq2)


def calculate_freq(n_freq: int, freq_start: int, toler: float, elements: List[Element], n_glb_dof: int) -> List[float]:
    """
    计算从第 freq_start 阶开始的共 n_freq 个圆频率（rad/s）。

    返回:
    - 长度为 n_freq 的圆频率列表
    """
    res: List[float] = []
    for k in range(freq_start, freq_start + n_freq):
        res.append(calculate_kfreq(k, toler, elements, n_glb_dof))
    return res

#仅进行轴向振动，所以求j0那里需要去掉Jb，仅保留Ja
if __name__ == "__main__":
    E = 1.0
    I = 1.0
    rho = 1.0
    A = 1.0
    L = 1.0
    #轴向振动悬臂杆件仅有轴向位移
    elem1 = Element(Len=L*1, EA=E * A, EI=E * I, mass=rho * A, CosA=1.0, SinA=0.0, GlbDOF=[0, 0, 0, 1, 0, 0])
    elem2 = Element(Len=L*2, EA=E * A, EI=E * I, mass=rho * A, CosA=1.0, SinA=0.0, GlbDOF=[1, 0, 0, 2, 0, 0])
    n_glb_dof = 2
    freqs = calculate_freq(20, 1, 1e-4, [elem1,elem2], n_glb_dof)
    for i, w in enumerate(freqs, 1):
        print(f"{i}: {w:.4f}",f"理论值: {(2*i-1)/6*math.pi:.4f}")

