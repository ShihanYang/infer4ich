# read wordset vectors and Disciplinary Vocabulary Vector Spaces from .csv file
# and computing the projection of the wordset vectors onto the disciplinary space.

from projection import project_onto_orthogonal_basis as project
from similarity import wasserstein_distance as wasserstin
from similarity import hausdorff_distance as hausdorff
import numpy as np

def loadEmbedding(file):
    '''
    file: a embedding .csv file name
    return: 
    '''
    embeddings = list()
    with open(file, 'r', encoding='utf-8') as wsf:
        lines = wsf.readlines()
        for line in lines:
            record = line.split('@')
            vec = eval(record[1].strip())
            embeddings.append(vec)        
    embeddings = np.array(embeddings)
    # print('  Got embeddings:', embeddings.shape)
    return embeddings


def project_vector_set(wordset_embeddings, space_embeddings):
    '''
    wordset_embeddings: the set of vectors of all wordset
    space_embeddings: the vector space being projected
    return: a projection matrix with one projection vector per row
    '''
    projs = list()
    for vec in wordset_embeddings:
        proj = project(vec, space_embeddings)
        projs.append(proj)
    projs = np.array(projs)
    return projs


def score(vectors):
    '''
    求投影矩阵的最大奇异值，然后变换出得分
    vectors: a matrix with row-main
    return: score
    '''
    m = vectors.shape[0]
    n = vectors.shape[1]
    U, S, V = np.linalg.svd(vectors)
    tolerance = max(np.finfo(vectors.dtype).eps * max(vectors.shape) * S[0], 1e-10)
    rank = np.sum(S > tolerance)
    # sv = max(S) / min(vectors.shape)   # rank   # np.sqrt(n)  # 最大奇异值小于等于min(m,n)
    sv = max(S) * np.sqrt(rank) / np.sqrt(n)
    return sv
    
    
def score2(vectors):
    '''
    计算投影的平均长度来进行比较。
    用L1范数计算长度。注意：实验表明，不大容易区分两组投影向量，与向量各维度的分布和向量的个数有关
    不能用余弦相似性，到同一空间的投影都平行
    '''    
    number = vectors.shape[0]  # 行向量个数
    overall_length = 0.0
    sv = 0.0
    for row in vectors:
        overall_length += sum(row)  # np.linalg.norm(row) 
    sv = overall_length / number  # 平均长度
    # print('  ', overall_length, number, sv)
    # 规范化到[0,1]区间，sigmoid函数
    k = 1.2  # k越大sigmoid函数越陡峭
    sigmoid_sv = 1 / (1 + np.exp(-k * sv))
    return sigmoid_sv


if __name__ == '__main__':

    wordset_vec = list()  # 词集中每个单词的向量

    space_vec = list()  # 学科词汇空间所有的向量
    
    wordset_file = 'data/bai-syj.csv.vec'
    wordset_file_2 = 'data/hani-jzsl.csv.vec'
    space_file = 'data/Sociology.csv.vec'

    wordset_vec = loadEmbedding(wordset_file)
    space_vec = loadEmbedding(space_file)

   
    ############################################################
    # 用全部向量计算投影，然后求投影矩阵的得分
    ############################################################
    projections = project_vector_set(wordset_vec, space_vec)
    
    wordset_vec2 = loadEmbedding(wordset_file_2)
    projections2 = project_vector_set(wordset_vec2, space_vec)
    
    # 归一化
    space_vec = space_vec / (np.linalg.norm(space_vec, axis=1, keepdims=True) + 1e-10)
    wordset_vec = wordset_vec / (np.linalg.norm(wordset_vec, axis=1, keepdims=True) + 1e-10)
    wordset_vec2 = wordset_vec2 / (np.linalg.norm(wordset_vec2, axis=1, keepdims=True) + 1e-10)
    
    print(wordset_file, 'Score is', score(projections), 'in', space_file)
    print(wordset_file_2, 'Score is', score(projections2), 'in', space_file)  
    
    print(wordset_file, 'Score-2 is', score2(projections), 'in', space_file)
    print(wordset_file_2, 'Score-2 is', score2(projections2), 'in', space_file)  
    
    # print('distance themselves:', wasserstin(wordset_vec, wordset_vec2))
    # print('distance in the same space:', wasserstin(projections, projections2))
    # print('distance p1 and space:', wasserstin(wordset_vec, space_vec))  
    # print('distance p2 and space:', wasserstin(wordset_vec2, space_vec))
    
    print('hausdorff distance themselves:', hausdorff(wordset_vec, wordset_vec2))
    print('hausdorff in the same space:', hausdorff(projections, projections2))
    # print('hausdorff p1 and space:', hausdorff(wordset_vec, space_vec))  
    # print('hausdorff p2 and space:', hausdorff(wordset_vec2, space_vec))
   