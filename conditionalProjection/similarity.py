#  @file: similarity.py
#  @version：1.0.5
#  @brief: computing the similarity between vectors
#  @creation date: 2025.06.28
#  @last modified date: 2026.4.22 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license.


import numpy as np
import projection as prj

# Cosine similarity 
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    simi = dot_product / (norm_a * norm_b)
    return  simi  # (simi + 1) / 2  # 限制范围到[0, 1]


# Euclidean Distance
def euclidean_similarity(vec1, vec2):
    dist = np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))
    similarity = 1 / (1 + dist)  # 转换为相似度，没多大意义，只是变换到小于1而已
    return similarity


# Dot Product Similarity
def dot_product_similarity(vec1, vec2):
    v1 = vec1 / sum(vec1)   # 向量归一化 
    v2 = vec2 / sum(vec2)   
    return np.dot(v1, v2)


# Pearson Correlation
def pearson_similarity(vec1, vec2):
    from scipy.stats import pearsonr
    corr, _ = pearsonr(vec1, vec2)   # corr范围[-1, 1]
    return corr


# Manhattan Distance
def manhattan_similarity(vec1, vec2):
    from scipy.spatial.distance import cityblock
    distance = cityblock(vec1, vec2)
    return distance


# Conditional Similarity
def conditional_similarity(vec1, vec2, matrix):
    '''
    计算两个向量在basis张成的向量空间上的相似性
    vec1, vec2: m dimention
    matrix    : n rows m columns
    '''
    Q, R = np.linalg.qr(matrix) 
    basis = Q   # 空间的标准正交基
    # print(basis.shape)
    prj_vec1 = prj.project_onto_orthogonal_basis(vec1, basis)   # 投影计算要求basis是正交基
    prj_vec2 = prj.project_onto_orthogonal_basis(vec2, basis)
    # comp_vec1 = vec1 / sum(vec1) - prj_vec1  # projection 是归一化之后的，这里原向量先归一化再求补
    # comp_vec2 = vec2 / sum(vec2) - prj_vec2
    # print('  vectors:', vec1, vec2)
    # print('  projection:', prj_vec1, prj_vec2)
    # print('  comp_vecs:', comp_vec1, comp_vec2)
    # similarity = cosine_similarity(vec1, vec2)
    similarity_prj = cosine_similarity(prj_vec1, prj_vec2)
    # similiarity_comp = cosine_similarity(comp_vec1, comp_vec2)
    # print('  similarity_ori:', similarity)
    # print('  similarity_prj:', similarity_prj)
    # print('  similarity_cmp:', similiarity_comp)
    # print('  total (prj+cmp):', similarity_prj + similiarity_comp)
    return  similarity_prj  



#################################################################
# 计算两组不同个数的同维向量之间的Wasserstein距离
# 语义需求方面: 若关注整体形状相似性，用豪斯多夫或最大均值差异 (MMD)；
#              若关注分布，用Wasserstein或KL散度。
#################################################################
def wasserstein_distance(X, Y):
    import ot  # Python Optimal Transport库
    
    # 计算成本矩阵 (通常使用欧氏距离)
    M = ot.dist(X, Y)
    
    # 定义权重 (通常是均匀分布)
    one_x = X.shape[0]
    one_y = Y.shape[0]
    a = np.ones((one_x,)) / one_x
    b = np.ones((one_y,)) / one_y 

    # Sinkhorn近似，reg是正则化参数
    w_dist_approx = ot.sinkhorn2(a, b, M, reg=0.1)  # 用于计算两个概率分布之间的 Sinkhorn 距离（基于熵正则化的最优传输距离）
    # print(f"近似Wasserstein距离: {w_dist_approx}")
    return w_dist_approx

def hausdorff_distance(u, v):
    from scipy.spatial.distance import directed_hausdorff
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])



if __name__ == '__main__':
    
    vec1 = np.array([1, 2, 3, 1])
    vec2 = np.array([3, 5, 3, 1])

    matrix = np.array([[5, 5, 3, 2], [2.1, -2, 0, 1], 
                       [9, 0, 4, 1], [0.5,  1, 3, 8], 
                       [9, 3, 8, 0], [1,    2, 3, 1], 
                       [4, 2, 6, 1], [-1,   2, 0, 1],
                       [1, 2, 1, 4], [-7,  -2, 2, 4]]
                      )  # 行向量！下面求正交基不需要转置。
    

    # print(cosine_similarity(vec1, vec2))  
    # print(euclidean_similarity(vec1, vec2))  
    # print(dot_product_similarity(vec1, vec2)) 
    # print(pearson_similarity(vec1, vec2))
    # print(manhattan_similarity(vec1, vec2)) 

    print(conditional_similarity(vec1, vec2, matrix))  
