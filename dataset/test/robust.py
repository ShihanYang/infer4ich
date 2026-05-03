#  @file: classifying.py
#  @version:1.0.0
#  @brief: Testing the robustness of the KNN algorithm on imbalanced and balanced datasets.
#  @creation date: 2026.04.28
#  @last modified date: 2026.05.03 
#  @authors: S. Yang
#  @copyright: © 2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


## 针对F1分数进行独立样本t检验,核心在于将两组多次实验得到的F1分数视为两个独立的样本,通过比较它们的均值来判断差异是否显著。
## F1分数是精确率(Precision)和召回率(Recall)的调和平均数,它综合考量了模型在正类预测上的准确性与覆盖率。
## 特别是在不平衡样本中,单纯的精度(Accuracy)往往会虚高,而F1分数能更真实、敏感地反映模型的分类性能。因此,
## 用F1分数进行独立样本t检验,是证明算法在面对不同数据分布时依然保持性能稳定的最有力证据。
'''
1. 数据准备:将你在“不平衡数据集”上多次实验得到的F1分数作为样本1,在“平衡数据集”上多次实验得到的F1分数作为样本2。
2. 方差齐性检验(Levene检验):在进行t检验前,先判断两组数据的方差是否相等。
3. 执行独立样本t检验:根据方差齐性的结果,选择对应的t检验方式(标准t检验或Welch校正t检验)。
4. 结果解读:通过P值判断是否存在显著差异。

如果最终得到的 P值 ≥ 0.05:说明在统计学上,该算法在平衡数据集和不平衡数据集上的F1分数没有显著差异。你可以据此得出结论:算法对数据分布的平衡性不敏感,具有良好的稳定性。
如果 P值 < 0.05:说明两组F1分数存在显著差异。此时你可以进一步计算效应量(Cohen's d),来评估这种差异在实际应用中到底是大还是小。

'''


import numpy as np
from scipy import stats

from pathlib import Path
import sys
cwd = Path(__file__).parent
infer_d = cwd.parent.parent 
target_dir = str(infer_d) 
sys.path.append(target_dir)
from infer.classifying import *


# 1. 准备数据(替换为你实际多次实验得到的F1分数)
'''
针对对比平衡与不平衡数据集,最常用且科学的方法是自助法或重复K折交叉验证。
'''
from sklearn.model_selection import RepeatedStratifiedKFold
# 设置重复K折 (m折,重复n次 -> 得到m*n个结果)
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=200, random_state=48)

assessvalue = np.loadtxt(target_dir+'/infer/'+'assesmentvalue++.csv', dtype=float, delimiter=',')
X, y = assessvalue[:, [0, 2]], assessvalue[:, -1]
# print(X.shape, y.shape)   # (188, 4) (188,)
f1_scores_list_A = []
for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = KNNClassifier(K=7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_scores_list_A.append(f1_score(y_test, y_pred, average='micro'))  #  or 'micro', 'weighted'
print(f"Got {len(f1_scores_list_A)} F1-scores for imbalanced dataset.")


from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
# 平衡采样:SMOTE通过插值算法,在少数类样本之间生成新的“合成”样本,从而增加少数类的代表性。
#          欠采样,最简单的方法,随机删除多数类样本。
#          自助法bootstrap, 它是有放回的随机采样,每次采样的数据分布会有微小波动,非常适合用来模拟不同的数据集环境。
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X, y)   # (339, 4) (339,)
# rus = RandomUnderSampler(random_state=42)
# X_res, y_res = rus.fit_resample(X, y)   # (90, 4) (90,)
X_res, y_res = resample(X, y, replace=True, n_samples=188, random_state=48)  
# print(X_res.shape, y_res.shape)  # (188, 4) (188,)
f1_scores_list_B = []
for train_index, test_index in rkf.split(X_res, y_res):
    X_train, X_test = X_res[train_index], X_res[test_index]
    y_train, y_test = y_res[train_index], y_res[test_index]
    model = KNNClassifier(K=7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_scores_list_B.append(f1_score(y_test, y_pred, average='micro'))  #  or 'micro', 'weighted'
print(f"Got {len(f1_scores_list_B)} F1-scores for balanced dataset.")


f1_imbalanced = f1_scores_list_A   # 不平衡数据集的F1分数
f1_balanced   = f1_scores_list_B   # 平衡数据集的F1分数

# 2. 方差齐性检验 (Levene检验)
# 如果 p > 0.05,说明两组方差相等(方差齐性)
levene_stat, levene_p = stats.levene(f1_imbalanced, f1_balanced)
print(f"Levene检验 P值: {levene_p:.6f}")

# 3. 独立样本t检验
# equal_var 参数根据 Levene 检验的结果来设定
# 如果方差齐 (levene_p > 0.05),则 equal_var=True；否则 equal_var=False (即使用 Welch's t-test)
equal_var = levene_p > 0.05
t_stat, p_value = stats.ttest_ind(f1_imbalanced, f1_balanced, equal_var=equal_var)

print(f"t统计量: {t_stat:.6f}")
print(f"独立样本t检验 P 值: {p_value:.6f}")


# 如果 P值 < 0.05:说明两组F1分数存在显著差异。此时你可以进一步计算效应量(Cohen's d),来评估这种差异在实际应用中到底是大还是小。
def calculate_cohens_d(group1, group2):
    # 1. 计算均值
    m1, m2 = np.mean(group1), np.mean(group2)
    
    # 2. 计算方差 (ddof=1 表示无偏估计)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 3. 获取样本量
    n1, n2 = len(group1), len(group2)
    
    # 4. 计算合并标准差 (Pooled Standard Deviation)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # 5. 计算 Cohen's d
    d = (m1 - m2) / pooled_std
    
    return d

d_value = calculate_cohens_d(f1_balanced, f1_imbalanced)
print(f"Cohen's d: {d_value:.4f}")

'''
Cohen's d 值   效应等级	      解读 
0.0 - 0.19	   可忽略	  差异极小,几乎可以认为两个模型表现一致。
0.2 - 0.49	   小效应	  差异存在,但可能在实际工程中不明显。
0.5 - 0.79	   中等效应	  差异适中,具有实际参考价值。
≥ 0.8	       大效应	  差异非常明显,算法受数据平衡性影响巨大。
'''

''' 2026-05-03
Got 1000 F1-scores for imbalanced dataset.
Got 1000 F1-scores for balanced dataset.
Levene检验 P值: 0.309586
t统计量: -3.751938
独立样本t检验 P 值: 0.000180
Cohen's d: 0.1678
'''
