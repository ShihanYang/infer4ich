#  @file: classifying2.py
#  @version：1.0.5
#  @brief:  This file is used to implement a k-means classifier and plot the distribution of three classes of sample points.
#           Note: modify the input filename to load different datasets. 
#  @creation date: 2025.08.28
#  @last modified date: 2026.04.23 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from pathlib import Path
pwd = Path(__file__).resolve().parent

file = pwd / 'assesmentvalue++.csv'  # modify the input filename to load different datasets
ori_assessvalue = np.loadtxt(file, dtype=float, delimiter=',')   
# print(type(assessvalue), assessvalue.shape, assessvalue[0:5])  
assessvalue = ori_assessvalue[:, 0:5]  # Use the first five columns for clustering
labels = ori_assessvalue[:, 5]  # Use the sixth column for labels

# Use the elbow method to determine the optimal value of k in k-means clustering.
sse = []
k_range = range(1, 13)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=68)
    km.fit(assessvalue)
    sse.append(km.inertia_) # inertia_ is SSE (Sum of Squared Errors)

# Smooth the SSE curve using cubic spline interpolation
x = np.array(k_range)
y = np.array(sse)
x_smooth = np.linspace(x.min(), x.max(), 400) 
y_smooth = make_interp_spline(x, y, k=3)(x_smooth)

plt.figure(figsize=(6, 4.2))
plt.plot(x_smooth, y_smooth, color='green')
plt.plot(x, y, 'o',markeredgecolor='red', markerfacecolor='red')
# plt.plot(k_range, sse, color='green', marker='o', markeredgecolor='red', markerfacecolor='red')
plt.xlabel('k')
plt.ylabel('SSE')  
plt.title('Elbow Method For Optimal k in k-means Clustering')
# plt.savefig(pwd / 'elbow.svg')
# plt.show()  # The visualization of the data shows a clear elbow around k = 3 ！


# the optimal k value is 3
k = 3  
seeds = assessvalue[np.random.choice(assessvalue.shape[0], k, replace=False)]
# print(seeds)
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
# print(c3, np.array(c3).shape)  

# decide the label of each class
pred_labels = []
c1_0, c1_1, c1_2 = 0, 0, 0
c2_0, c2_1, c2_2 = 0, 0, 0
c3_0, c3_1, c3_2 = 0, 0, 0
epsilon = 1e-4
for item in ori_assessvalue:
    if any(np.allclose(item[0:5], x, atol=epsilon) for x in c1):
        pred_labels.append('c1')
        if item[5] == 0: c1_0 += 1
        elif item[5] == 1: c1_1 += 1
        elif item[5] == 2: c1_2 += 1
    elif any(np.allclose(item[0:5], x, atol=epsilon) for x in c2):
        pred_labels.append('c2')
        if item[5] == 0: c2_0 += 1
        elif item[5] == 1: c2_1 += 1
        elif item[5] == 2: c2_2 += 1
    elif any(np.allclose(item[0:5], x, atol=epsilon) for x in c3):
        pred_labels.append('c3')
        if item[5] == 0: c3_0 += 1
        elif item[5] == 1: c3_1 += 1
        elif item[5] == 2: c3_2 += 1
label_c1 = np.argmax([c1_0, c1_1, c1_2])  # voting for the label of each class
label_c2 = np.argmax([c2_0, c2_1, c2_2])
label_c3 = np.argmax([c3_0, c3_1, c3_2])
print(f'label of each class: c1:{label_c1}, c2:{label_c2}, c3:{label_c3}')

for i in range(len(pred_labels)):
    if pred_labels[i] == 'c1':
        pred_labels[i] = label_c1
    elif pred_labels[i] == 'c2':
        pred_labels[i] = label_c2
    elif pred_labels[i] == 'c3':
        pred_labels[i] = label_c3

# [0:'goverment-led', 1:'community-led', 2:'mixed pattern']
def num2type(num):
    if num == 0:
        return "governmend-led"
    elif num == 1:
        return "community-led"
    else:
        return "mixed pattern"
    
label_c1 = num2type(label_c1)
label_c2 = num2type(label_c2)
label_c3 = num2type(label_c3)

# print(pred_labels)
# print(labels)
print('Accuracy:', np.sum(pred_labels == labels) / len(labels))
print('  total samples:', len(pred_labels))
print('  true numbers:', accuracy_score(labels, pred_labels, normalize=False))
print('  accuracy_score:', accuracy_score(labels, pred_labels))
print('  F1-score:', f1_score(labels, pred_labels, average='weighted'))
print(classification_report(labels, pred_labels))
# Reporting confusion matrix
cm = confusion_matrix(labels, pred_labels)
print("Confusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=[f'Class 1: {label_c1}', f'Class 2: {label_c2}', f'Class 3: {label_c3}'])
disp.plot(cmap='Reds')
# plt.show()

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
