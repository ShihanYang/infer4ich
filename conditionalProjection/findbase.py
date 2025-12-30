# Find the orthogonal basis of a big matrix

import numpy as np
import ast

vectors = list()

with open("data/embedding.csv", 'r', encoding='utf-8') as mf:
    lines = mf.readlines()
    for line in lines:
        data = line.strip().split('@')  # Note: different seperator
        # print(len(data), type(data))  
        # print(len(data[1]), type(data[1]))
        vec = ast.literal_eval(data[1])  # transform the string '[1,2,3]' into a list [1,2,3]
        # print(len(vec), type(vec), vec[:5])
        vectors.append(vec)
        
print(len(vectors))
bigmatrix = np.array(vectors)
print(bigmatrix.shape)

# 求矩阵的正交基
# QR分解的Q矩阵的前r列就是一组正交基，其中r是秩
# QR分解（QR Decomposition）是一种将矩阵分解为正交矩阵（Q）和上三角矩阵（R）的线性代数方法。QR分解本质上是将矩阵的列空间用一组正交基表示。
Q, R = np.linalg.qr(bigmatrix.T)  # 要转置
rank = np.linalg.matrix_rank(R)  # same as matrix_rank(Q)
print(Q.shape, R.shape, rank)  
Q = Q.T 
print('Orthogoal basis for', bigmatrix.shape, 'is:', Q.shape, Q[:2])
