import numpy as np
import math

def get_reduced_stiffness_matrix(E, I, L):
    """
    计算两端简支欧拉-伯努利梁的简化刚度矩阵。
    参数:
    E:  弹性模量
    I : 截面惯性矩
    L : 梁的长度
    返回:
    np.ndarray: 简化后的刚度矩阵 (2x2)
    """
    factor = E * I / L
    K_reduced = factor * np.array([
        [4, 2],
        [2, 4]
    ])
    return K_reduced

def get_reduced_dynamic_stiffness_matrix(E, I, L, rho, A, f):
    """
    计算给定频率下的动力刚度矩阵。
    参数:
    E (float): 弹性模量
    I (float): 截面惯性矩
    L (float): 梁的长度
    rho (float): 材料密度
    A (float): 截面积
    f (float): omega  w

    返回:
    np.ndarray: 简化后的动力刚度矩阵 (2x2)
    """
    # 简化刚度矩阵
    K_reduced = get_reduced_stiffness_matrix(E, I, L)

    # 简化质量矩阵
    mass_factor = rho * A * L**3 / 420
    M_reduced = mass_factor * np.array([
        [4, -3],
        [-3, 4]
    ])
    # 动力刚度矩阵
    Kd_reduced = K_reduced - f**2 * M_reduced
    return Kd_reduced

def gaussian_elimination(matrix):
    """
    通过高斯消元将矩阵转换为上三角形式。
    此实现包含部分主元法以提高数值稳定性。

    参数:
        A (numpy.ndarray): 输入的方阵。

    返回:
        numpy.ndarray: 上三角矩阵。
    """
    # 创建一个副本以避免修改原始矩阵
    mat = np.array(matrix, dtype=float)
    n = mat.shape[0]

    for i in range(n):
        # 根据要求，不进行主元选择（换行）
        pivot = mat[i, i]
        if np.isclose(pivot, 0)<1e-6:
            # 如果主元为零，则跳过此列的消元
            continue

        for j in range(i + 1, n):
            factor = mat[j, i] / pivot
            mat[j, i:] -= factor * mat[i, i:]

    return mat

def count_negative_diagonal_elements(matrix):
    """
    计算矩阵对角线上负元素的个数。
    参数:
    matrix (np.ndarray): 输入的方阵
    返回:
    int: 对角线上负元素的数量
    """
    count = 0
    for i in range(len(matrix)):
        # 主对角线元素：matrix[i][i]
        if matrix[i][i] < 0:
            count += 1
    return count

def calculate_j0(freq) -> int:
    """计算J0值(改进版本)"""

    L = 1.0
    EA =1.0
    EI =1.0
    m =1.0

    # 轴向振动特征值个数
    nu =freq * L * math.sqrt(m / EA)
    ja = int(nu /math.pi)

    # 弯曲振动特征值个数（当 EI<=0 时表示无弯曲刚度，直接不计入）
    if EI <= 0:
        jb = 0
    else:
        lambda_val = L * (freq ** 2 * m / EI) ** 0.25

        if lambda_val > 1e-6:
            inv_e = math.exp(-lambda_val)
            cos_lambda = math.cos(lambda_val)

            # 改进的符号判断
            discriminant = inv_e - cos_lambda * (1.0 + inv_e ** 2) / 2.0
            sg = 1 if discriminant > 0 else -1

            n_lambda = int(lambda_val / math.pi)
            jb = n_lambda - (1 - (-1) ** n_lambda * sg) /2
        else:
            jb = 0
    # jb：单元固端的横向弯曲振动频率。结构两端简支，弯曲振动位移始终为零，故jb=0，不计入
    j0 = ja
    return max(0, j0)    # 确保非负

def  calculate_JK(freq):
    M=get_reduced_dynamic_stiffness_matrix(1,1, 1, 1, 1, freq)
    M_1=gaussian_elimination(M)
    num=count_negative_diagonal_elements(M_1)
    return num

for i in range(1,21,1):
    kfreq = i
    freq_1 = 1
    freq_2 =10
    while True:
        J0_1 = calculate_j0(freq_1)
        Jk_1 = calculate_JK(freq_1)
        J_total = J0_1 + Jk_1
        if J_total < kfreq:
            break
        freq_1 = freq_1 / 1.5
        if freq_1 < 1e-12:     # 防止频率过小导致死循环
            break
    while True:
        J0_2 = calculate_j0(freq_2)
        Jk_2 = calculate_JK(freq_2)
        J_total = J0_2 + Jk_2
        if J_total >= kfreq:
            break
        freq_2 = freq_2 * 1.2
        if freq_2 > 1e12:      # 防止频率过大导致死循环
            break
    iteration = 0
    while True:
        iteration += 1
        freq = (freq_1 + freq_2) / 2.0
        J0 = calculate_j0(freq)
        Jk = calculate_JK(freq)
        J_total = J0 + Jk

        if J_total >= kfreq:
            freq_2 = freq
        else:
            freq_1 = freq

        if (freq_2 - freq_1) <= 0.00001 * (1.0 + freq_2):   #收敛精度
            break
        if iteration > 150:      # Safety break
            break
    final_freq = (freq_1 + freq_2) / 2.0
    print(f'第{i}阶w：',  f"{final_freq ** 2:.4f}",'j0和jk：',J0,',',Jk)










