#  @file: my_module.py
#  @version：1.1.0
#  @brief: Compute conditional similarity of two projections onto multi-spaces
#    - two projection vector sets are saved in the .csv.vec files
#    - multi-spaces are already weighted by AHP method
#    - there are many methods to compute the similarity of two vector sets
#  @creation date: 2025.05.28
#  @last modified date: 2026.04.22 
#  @authors: S. Yang
#  @copyright: © 2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 

import numpy as np
from datetime import datetime as dt
from projection2 import loadEmbedding, project_vector_set, score
from yaspin import yaspin, Spinner


# weights for each space (history, aesthetic, semiology, sociology)  
# TODO: These weights can be changed when getting more evidence
on_independ_interdepend = (0.2949, 0.0475, 0.1471, 0.5104)
on_individ_collect = (0.2953, 0.0509, 0.1311, 0.5227)
on_tight_loose = (0.2760, 0.1001, 0.0917, 0.5322)
on_relmobility = (0.5573, 0.0485, 0.1415, 0.2527)
weights = {
    'independ_interdepend' : (0.2949, 0.0475, 0.1471, 0.5104),
    'individ_collect' : (0.2953, 0.0509, 0.1311, 0.5227),
    'tight_loose' : (0.2760, 0.1001, 0.0917, 0.5322),
    'relmobility' : (0.5573, 0.0485, 0.1415, 0.2527)
}
on_spaces = [(x+y+z+r)/4 for x,y,z,r in zip(on_independ_interdepend, on_individ_collect, on_tight_loose, on_relmobility)]


#################################################################
# 计算两组不同个数的同维向量之间的Wasserstein距离
# 语义需求方面: 若关注整体形状相似性，用豪斯多夫或最大均值差异 (MMD)；
#              若关注分布，用Wasserstein或KL散度。
#################################################################
def wasserstein_distance(X, Y):
    '''
    X, Y: ndarray 数据类型，二维矩阵，它们的行数不同，列数相同
    '''
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

#################################################################
# 计算两组不同个数的同维向量之间的Hausdorf距离
#################################################################
def hausdorff_distance(u, v):
    from scipy.spatial.distance import directed_hausdorff
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

#################################################################
# 计算两组向量之间的规范化距离
#################################################################
def normalized_distance(u, v):
    '''
    u, v: a set of vectors with same dimension. 
    Note: If u and v are already normalized, their L2 norm is always 1. 
          So, the distance between u and v is always 0.
    '''
    u_total = 0
    v_total = 0
    for i in u:
        u_total += np.linalg.norm(i)
    for i in v:
        v_total += np.linalg.norm(i)
    dist = np.abs(u_total/u.shape[0] - v_total/v.shape[0])
    return dist

#################################################################
# 计算两组向量之间的最大特征值距离
#################################################################
def maxeigen_distance(u, v):
    _, S1, _ = np.linalg.svd(u)
    _, S2, _ = np.linalg.svd(v)
    dist = max(S1) - max(S2)  
    return dist

def sigmoid(value):
    sig = 1.0 / (1.0 + np.exp(-value))
    return sig

#################################################################
# 根据距离来计算两组向量直接的相似性
#################################################################
def projections_similarity(distance):
    return 1 - sigmoid(distance)

#################################################################
# 在四个心理认知维度上分别计算条件相似性
#################################################################
def psychology_similarity(one, two):
    '''
    one: 存放第一个对象的四个认知维度的 .csv.vec embedding files的目录
    two: 存放第一个对象的四个认知维度的 .csv.vec embedding files的目录
    return: Weighted four conditional similarity 
            on all spaces for each dimension of psychology
    '''
    dimensionalSimilarity = dict()
    
    print(f'@ {dt.now()} - pyschology dimension similarity between {one.upper()} and {two.upper()}')
    
    # 1. load spaces embeddings
    spaces = dict()
    path = r"F:/mycodes/wordev/data/"    
    space_files = [
        "History.csv.vec",
        "Aesthetic.csv.vec",
        "Semiology.csv.vec",
        "Sociology.csv.vec"    
    ]
    print("loading space embeddings ...")
    for sf in space_files:
        key = sf.split('.')[0].lower()
        spaces[key] = loadEmbedding(path + sf)
    print('& Loaded.')
        
    # 2. load vector sets embeddings
    vsembeddings = dict()
    vsembedding_files = [one+'/independ_interdepend.csv.vec', one+'/individ_collect.csv.vec',
                         one+'/tight_loose.csv.vec', one+'/relmobility.csv.vec',
                         two+'/independ_interdepend.csv.vec', two+'/individ_collect.csv.vec',
                         two+'/tight_loose.csv.vec', two+'/relmobility.csv.vec']
    vse_name = list()  # len(vse_name) = 8
    print("loading wordset embeddings ...")
    for wsf in vsembedding_files:
        key = wsf.split('.')[0].lower()
        vse_name.append(key)
        vsembeddings[key] = loadEmbedding(path + wsf)
    print("& Loaded.")
    
    # 3. compute projections onto each space
    projections = dict()
    spacename = ['history', 'aesthetic', 'semiology', 'sociology'] 
    lines = ['--', '\\\\', '||', '//']  # for long performing animation effects
    with yaspin(Spinner(lines, interval=8), 
                text="  is processing", 
                ellipsis='...', 
                color="blue", 
                side='right', 
                timer=True) as spinner:
        spinner.write("projecting ... ")
        # every one is projected onto each space
        for space in spacename:
            for vse in vse_name:
                projections[(vse, space)] = project_vector_set(vsembeddings[vse], spaces[space]) 
        spinner.ok("& OK: Projected.")
            
    # 4. compute the distance of pairwise projection on each space
    distanceOn = dict()  # distance between two sets of projections in each space 
    dimension = ['independ_interdepend', 'individ_collect', 'tight_loose', 'relmobility']
    for space in spacename:
        for dim in dimension:
            x = projections[(one+'/'+dim, space)]
            y = projections[(two+'/'+dim, space)]
            dist = maxeigen_distance(x, y)
            print('  their distance', dim, 'in', space, ':', dist)
            distanceOn[(dim, space)] = dist
    
    # 5. compute the similarity of pairwise projection 
    # Perform distance transformation into [-3, 3] from [0, 1], y = 6x - 3
    for dim in dimension:
        group = {k:v for k,v in distanceOn.items() if k[0] == dim}
        exptotal = sum(np.exp(list(group.values())))
        for key in group.keys():  # softmax
            distanceOn[key] = np.exp(distanceOn[key]) / exptotal 
            distanceOn[key] = 6 * distanceOn[key] - 2.9
    similarityOn = dict()
    for dist in distanceOn.keys():  # sigmoid
        similarityOn[dist] = projections_similarity(distanceOn[dist]) 
    
    # 6. weighted on all spaces
    print('Dimensional similarity on each space:')
    for dim in dimension:
        group = {k:v for k,v in similarityOn.items() if k[0] == dim}
        for space in spacename:
            print('  similarity', dim, 'in', space, ':', group[(dim,space)])
        print('weights vector in', dim, weights[dim])
        weighted_similarity = np.dot(weights[dim], list(group.values()))
        print('Weighted dimensional similarity on all spaces:', weighted_similarity)
        dimensionalSimilarity[dim] = weighted_similarity
        
    print(f'&@ {dt.now()} DONE.')
    return dimensionalSimilarity
    

if __name__ == '__main__':
    
    ###############################################################
    # Perform the similarity between ICH projects one and another
    ###############################################################
    one = 'bai-syj'  # change data directory, 'lisu-dgj' or 'hani-jzsl'
    another = 'lisu-dgj'
    print(f'@ {dt.now()} - similarity between {one.upper()} and {another.upper()}')
    
    # 1. load spaces embeddings
    spaces = dict()
    path = r"F:/mycodes/wordev/data/"    
    space_files = [
        "History.csv.vec",
        "Aesthetic.csv.vec",
        "Semiology.csv.vec",
        "Sociology.csv.vec"    
    ]
    print("loading space embeddings ...")
    for sf in space_files:
        key = sf.split('.')[0].lower()
        spaces[key] = loadEmbedding(path + sf)
    print('& Loaded.')
        
    # 2. load vector sets embeddings
    vsembeddings = dict()
    vsembedding_files = ['bai-syj.csv.vec', 'hani-jzsl.csv.vec']
    vse_name = list()  # len(vse_name) = 2
    print("loading wordset embeddings ...")
    for wsf in vsembedding_files:
        key = wsf.split('.')[0].lower()
        vse_name.append(key)
        vsembeddings[key] = loadEmbedding(path + wsf)
    print("& Loaded.")
    
    # 3. compute projections onto each space, and cost too much here
    projections = dict()
    spacename = ['history', 'aesthetic', 'semiology', 'sociology'] 
    lines = ['--', '\\\\', '||', '//']  # for long performing animation effects
    with yaspin(Spinner(lines, interval=8), 
                text="  is processing", 
                ellipsis='...', 
                color="yellow", 
                side='right', 
                timer=True) as spinner:
        spinner.write("projecting ... ")
        # every one is projected onto each space
        for space in spacename:
            for vse in vse_name:
                projections[(vse, space)] = project_vector_set(vsembeddings[vse], # [:np.random.randint(10,21)], 
                                                               spaces[space])  # TODO Sometimes small data is used for testing, spaces[space][:120]
        spinner.ok("& OK: Projected.")
            
    # 4. compute the distance of pairwise projection on each space
    distanceOn = dict()  # distance between two sets of projections in each space 
    for space in spacename:
        x = projections[(vse_name[0], space)]
        y = projections[(vse_name[1], space)]
        # dist = wasserstein_distance(x, y)  # TODO: can be changed to other distance functions
        # dist = hausdorff_distance(x, y)
        # dist = normalized_distance(x, y)  # normalized distance
        dist = maxeigen_distance(x, y)
        print('  their distance in the', space, ':', dist)
        distanceOn[space] = dist
    
    # 5. compute the similarity of pairwise projection 
    # 5.5 Perform distance transformation into [-3, 3] from [0, 1], y = 6x - 3
    exptotal = sum(np.exp(list(distanceOn.values())))
    for key in distanceOn.keys():
        distanceOn[key] = np.exp(distanceOn[key]) / exptotal  # 因为可能含有负数，用Softmax函数法变换成随机向量，pi = exp(pi) / sum(exp(pi))
        distanceOn[key] = 6 * distanceOn[key] - 2.9
        
    similarityOn = dict()
    for dist in distanceOn.keys():
        similarityOn[dist] = projections_similarity(distanceOn[dist]) 
    
    # 6. weighted on all spaces
    print('Similarity on each space:')
    for simi in similarityOn.keys():
        print('  similarity in the', simi, ':', similarityOn[simi])
    print('weights vector :', on_spaces)
    weighted_similarity = np.dot(on_spaces, list(similarityOn.values()))
    print('Weighted similarity on all spaces:', weighted_similarity)
    
    print(f'&@ {dt.now()} DONE.')
    