#  @file: classifying2.py
#  @version：1.0.5
#  @brief:  该文件用于实现KNN分类器，并绘制三类样本点的分布
#           注意修改读入的文件名，以载入不同的数据集
#  @creation date: 2025.08.28
#  @last modified date: 2025.11.13 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


import numpy as np
import matplotlib.pyplot as plt

# TODO: 修改文件名，载入不同的数据集
assessvalue = np.loadtxt(r'assesmentvalue+.csv', dtype=float, delimiter=',')   # value+ 有五列数据
print(assessvalue.shape, assessvalue[0:5])  

k = 3  # k-means算法的k值，即分类数3
seeds = assessvalue[np.random.choice(assessvalue.shape[0], k, replace=False)]
print(seeds)
epochs = 10000

c1 = list()
c2 = list()
c3 = list()
for e in range(epochs):
    if e > 0:
        seeds = [np.mean(c1), np.mean(c2), np.mean(c3)]
        c1 = list()
        c2 = list()
        c3 = list()
    for v in assessvalue:
        distances = [np.linalg.norm(v - seed) for seed in seeds]
        k = distances.index(min(distances))
        # print(distances, k, min(distances))
        if k==0:  c1.append(v)
        if k==1:  c2.append(v)
        if k==2:  c3.append(v)


print('number in each class:', len(c1), len(c2), len(c3))
# print(c1)
# print(c2)
# print(c3, np.array(c3).shape)  # (108, 5)

# 绘制三类样本点的分布
if len(assessvalue.shape) == 1:
    plt.scatter(c1, [0]*len(c1), c='r', marker='o')
    plt.scatter(c2, [0]*len(c2), c='g', marker='o')
    plt.scatter(c3, [0]*len(c3), c='b', marker='o')
else:
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes[0,0].scatter(np.array(c1)[:,0], np.array(c1)[:,1], c='r', marker='o')
    axes[0,0].scatter(np.array(c2)[:,0], np.array(c2)[:,1], c='g', marker='o')
    axes[0,0].scatter(np.array(c3)[:,0], np.array(c3)[:,1], c='b', marker='o')
    axes[0,0].set_title('A) Feature 1-2 (Historical and Aesthetic)')

    axes[0,1].scatter(np.array(c1)[:,0], np.array(c1)[:,2], c='r', marker='o')
    axes[0,1].scatter(np.array(c2)[:,0], np.array(c2)[:,2], c='g', marker='o')
    axes[0,1].scatter(np.array(c3)[:,0], np.array(c3)[:,2], c='b', marker='o')
    axes[0,1].set_title('B) Feature 1-3 (Historical and Semiotic)')

    axes[0,2].scatter(np.array(c1)[:,0], np.array(c1)[:,3], c='r', marker='o')      
    axes[0,2].scatter(np.array(c2)[:,0], np.array(c2)[:,3], c='g', marker='o')  
    axes[0,2].scatter(np.array(c3)[:,0], np.array(c3)[:,3], c='b', marker='o')
    axes[0,2].set_title('C) Feature 1-4 (Historical and Sociological)')

    axes[1,0].scatter(np.array(c1)[:,1], np.array(c1)[:,2], c='r', marker='o')
    axes[1,0].scatter(np.array(c2)[:,1], np.array(c2)[:,2], c='g', marker='o')
    axes[1,0].scatter(np.array(c3)[:,1], np.array(c3)[:,2], c='b', marker='o')
    axes[1,0].set_title('D) Feature 2-3 (Aesthetic and Semiotic)')

    axes[1,1].scatter(np.array(c1)[:,1], np.array(c1)[:,3], c='r', marker='o')
    axes[1,1].scatter(np.array(c2)[:,1], np.array(c2)[:,3], c='g', marker='o')
    axes[1,1].scatter(np.array(c3)[:,1], np.array(c3)[:,3], c='b', marker='o')
    axes[1,1].set_title('E) Feature 2-4 (Aesthetic and Sociological)')

    axes[1,2].scatter(np.array(c1)[:,2], np.array(c1)[:,3], c='r', marker='o')
    axes[1,2].scatter(np.array(c2)[:,2], np.array(c2)[:,3], c='g', marker='o')
    axes[1,2].scatter(np.array(c3)[:,2], np.array(c3)[:,3], c='b', marker='o')
    axes[1,2].set_title('F) Feature 3-4 (Semiotic and Sociological)')
    
    plt.tight_layout()
plt.show()
