#  @file: projection.py
#  @version：1.0.5
#  @brief: # 求一个向量在一组（正交）基上（向量空间）的投影，及其在正交补空间上的分量
#  @creation date: 2025.08.28
#  @last modified date: 2025.10.12 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


import numpy as np

def project_onto_orthogonal_basis(v, matrix):
    """
    计算一个向量 v 在正交基 basis 上的投影    
    v: 目标向量 (numpy数组)
    basis: 正交基列表 (包含numpy数组的列表)    
    """
    n = matrix.shape[0]
    projection = np.zeros_like(v, dtype=float)  
    matrix = matrix[np.random.permutation(n)]  # 矩阵行数远大于秩，按行随机排列一下
    Q, _ = np.linalg.qr(matrix)   # QR分解，Q是正交基
    # 判断Q的秩
    rank_Q = np.linalg.matrix_rank(Q)
    if rank_Q < 768:
        Q = matrix
    for u in Q:
        # 计算分量系数
        coefficient = np.dot(v, u) / np.dot(u, u)
        # 累加到投影向量
        projection += coefficient * u
    projection = projection / np.linalg.norm(projection, ord=2)   # 欧几里得归一化,如此归一化之后向量长度总为1
    return projection

def component_in_orthogonal_complement(v, basis):
    '''
    求向量在一个空间的正交补空间上的分量
    '''
    projection = project_onto_orthogonal_basis(v, basis)
    complement = v/sum(v) - projection
    return complement



if __name__ == '__main__':
    # 例子
    # v = np.array([3, -2, 4])
    
    # basis = np.array([[1, 0, -1.5], [1, -2, -1], [2.5, -1, 1]])  # 假设已确认正交

    # # 验证正交性 (可选)
    # # assert np.isclose(np.dot(basis[0], basis[1]), 0), "基不正交"

    # proj = project_onto_orthogonal_basis(v, basis)
    # print("投影向量:", proj)
    # print("补空间上的分量：", v - proj)  # component_in_orthogonal_complement(v, basis)
    
    vec1 = np.array([1, 2, 0])
    vec2 = np.array([2, 4, 7])

    matrix = np.array([[13, 22, 33], [2.5, -1.6, 1], [2.5, -1.6, 1], [2,3,4]])  # 每行是一个向量！
    Q, R = np.linalg.qr(matrix)  # QR 分解将一个矩阵分解为一个正交矩阵（Q）和一个上三角矩阵（R）的乘积。
    basis = Q  # matrix
    
    projection1 = project_onto_orthogonal_basis(vec1, basis)
    projection2 = project_onto_orthogonal_basis(vec2, basis)
    
    print(projection1, np.linalg.norm(projection1), sum(projection1))   
    print(projection2, np.linalg.norm(projection2), sum(projection2))  
    
    # 规范化到[0,1]区间，sigmoid函数
    k = 1.0  # k越大sigmoid函数越陡峭
    sigmoid_sv = 1 / (1 + np.exp(-k * sum(np.abs(projection1)))) 
    print('score:', sigmoid_sv)
    sigmoid_sv2 = 1 / (1 + np.exp(-k * sum(np.abs(projection2)))) 
    print('score:', sigmoid_sv2)
