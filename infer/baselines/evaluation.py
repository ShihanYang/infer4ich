# ============================================================
#  @file: evaluation.py
#  @version：1.0.0
#  @brief: Experimental codes including Unified 5-fold cross-validation 
#                                     & Multi-baseline comparison
#  @creation date: 2026.10.02
#  @last modified date: 2026.10.15 
#  @authors: S. Yang
#  @copyright: ©2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 
# ============================================================

import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import mode, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import scipy.stats as st
import jieba

from tool_ready import ( 
    text_to_vector,
    extract_bert_features,
    get_kg_embedding,
    call_llm_func,
)



# ============================================================
# 1. Data Ready
# ============================================================
# X_w2v: Word2Vec/GloVe 平均向量 (n_samples, dim)
# X_bert: BERT [CLS] 向量 (n_samples, 768)
# X_kg: TransE/R-GCN 图嵌入 (n_samples, dim)
# LLM zero-shot 的预测结果 llm_preds（长度 n_samples，或每个测试折单独调用）
data = pd.read_csv("data.csv", header=None, names=['id', 'ichname', 'description', 'label'])
n_samples = data.shape[0]  # 188 samples
n_features = 5   # The number of samples is relatively small. To avoid the sparsity problem, the number of features should not be too large.

features_data = pd.read_csv(r"../assesmentdata+.csv")
features_data.columns = features_data.columns.str.strip()
X_tuned = features_data.iloc[:,[2,3,4,5,6]].values.astype(float)  # 第五列是similarity scores 是计算的逻辑得分，作为一个综合特征
# print(X_tuned.shape)  # (188, 5)

y = data['label'].values
y = np.array([0 if i == 'A' else 1 if i == 'B' else 2 for i in y])

texts = data.ichname +' := '+ data.description  # description for each ich

# Preparation of feature data for the baseline model
X_w2v = [text_to_vector(texts, text) for text in texts]  # (188, 200)
X_bert = extract_bert_features(texts)  # (188, 768)

X_kg = []
triples_file = r"triples.tsv"
entity_embeddings, relation_embeddings, index = get_kg_embedding(path = triples_file)
for item in data.ichname:
    print(item)
    vec = entity_embeddings[index[item]]
    X_kg.append(vec)
X_kg = np.array(X_kg)
print(X_kg.shape)  # (188, 64)

# Balanced dataset indexing: down-sample the community-led (label=1) class to 37 samples
idx_gov = np.where(y == 0)[0]
idx_com = np.where(y == 1)[0]
idx_mix = np.where(y == 2)[0]
np.random.shuffle(idx_com)
idx_com_balanced = idx_com[:37]
idx_balanced = np.concatenate([idx_gov, idx_com_balanced, idx_mix])
np.random.shuffle(idx_balanced)
# print(len(idx_balanced), idx_balanced)   # 112 balanced samples


# ============================================================
# 2. 统一评估函数
# ============================================================
def evaluate_model(X, y, model, n_splits=5, n_repeats=10, random_state=68):
    """
    使用 RepeatedStratifiedKFold 评估模型。
    X: 特征矩阵或文本列表（如果模型是 Pipeline 支持文本）
    y: 标签
    model: 实现 fit/predict 的分类器
    n_splits: 交叉验证折数
    返回：字典，包含 accuracy, precision, recall, f1, weighted_f1 的均值和标准差
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    accs, precs, recs, f1s, wf1s = [], [], [], [], []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in rskf.split(X, y):
        if isinstance(X, np.ndarray):
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            # 假设 X 是 list，支持索引
            X_train = [X[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        _, _, wf1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        precs.append(p); recs.append(r); f1s.append(f1); wf1s.append(wf1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'precision_mean': np.mean(precs),
        'precision_std': np.std(precs),
        'recall_mean': np.mean(recs),
        'recall_std': np.std(recs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'weighted_f1_mean': np.mean(wf1s),
        'weighted_f1_std': np.std(wf1s),
        'accs': accs,  # 保存每次的准确率用于统计检验
        'all_y_true': np.array(all_y_true),
        'all_y_pred': np.array(all_y_pred),
    }


# ============================================================
# 3. 本文方法：KNN (K=7)
# ============================================================
def run_knn(X, y, k=7, n_splits=3, n_repeats=5):
    model = KNeighborsClassifier(n_neighbors=k, metric="minkowski")  # metric='euclidean')
    return evaluate_model(X, y, model, n_splits, n_repeats)

# ============================================================
# 4. 决策树
# ============================================================
def run_decision_tree(X, y, n_splits=5, n_repeats=10):
    model = DecisionTreeClassifier(random_state=68)
    return evaluate_model(X, y, model, n_splits, n_repeats)

# ============================================================
# 5. K-means 分类器（K-means + 多数投票）
# ============================================================
class KMeansClassifier:
    def __init__(self, n_clusters=3, random_state=68):
        self.n_clusters = n_clusters
        self.random_state = random_state
    def fit(self, X, y):
        self.km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.km.fit(X)
        labels = self.km.labels_
        self.cluster_to_class = {}
        for c in range(self.n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                self.cluster_to_class[c] = mode(y[mask], keepdims=True)[0][0]
            else:
                self.cluster_to_class[c] = 0
        return self
    def predict(self, X):
        clusters = self.km.predict(X)
        return np.array([self.cluster_to_class[c] for c in clusters])

def run_kmeans(X, y, n_splits=5, n_repeats=10):
    model = KMeansClassifier(n_clusters=3, random_state=68)
    return evaluate_model(X, y, model, n_splits, n_repeats)

# ============================================================
# 6. TF-IDF + SVM
# ============================================================
def jieba_tokenizer(text):
    return list(jieba.cut(text))

def run_tfidf_svm(texts, y, n_splits=5, n_repeats=10):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None, max_features=5000, ngram_range=(1,2))),
        ('svm', LinearSVC(random_state=68, dual=True))
    ])
    return evaluate_model(texts, y, pipeline, n_splits, n_repeats)

# ============================================================
# 7. Word2Vec (/GloVe) + SVM
# ============================================================
def run_w2v_svm(X_w2v, y, n_splits=5, n_repeats=10):
    model = SVC(kernel='rbf', random_state=68)
    return evaluate_model(X_w2v, y, model, n_splits, n_repeats)

# ============================================================
# 8. BERT (/tunned) + Logistic Regression
# ============================================================
def run_bert_lr(X_bert, y, n_splits=5, n_repeats=10):
    model = LogisticRegression(max_iter=1000, random_state=68)
    return evaluate_model(X_bert, y, model, n_splits, n_repeats)

# ============================================================
# 9. LLM zero-shot 分类
# ============================================================
def llm_zero_shot_predict(texts, call_llm_func):
    """
    对每个文本调用 LLM, 返回预测标签数组。
    假设 LLM 输出 'A', 'B', 'C'，分别映射到 0, 1, 2。
    Note: Given the RMB cost implications, please execute this function with caution.
    """
    label_map = {'A': 0, 'B': 1, 'C': 2}
    preds = []
    for text in texts:  # it is basically one Chinese cent per query！
        prompt = f"""你是一个非物质文化遗产保护专家。请阅读以下项目描述，判断最适合的保护策略类型。
                    类别：
                    A. 政府主导
                    B. 社区主导
                    C. 混合型
                    只输出 A、B 或 C。

                    项目描述：
                    {text}
        """
        output = call_llm_func(prompt).strip().upper()
        # 简单解析，取第一个出现的 A/B/C
        pred = None
        for ch in output:
            if ch in label_map:
                pred = label_map[ch]
                break
        if pred is None:
            pred = 1  # 默认社区主导
        preds.append(pred)
    print("LLM zero-shot predictions:", preds)
    return np.array(preds)

# ============================================================
# 10. TransE (/R-GCN) 图嵌入 + KNN
# ============================================================
def run_kg_knn(X_kg, y, n_splits=5, n_repeats=10):
    model = KNeighborsClassifier(n_neighbors=7)
    return evaluate_model(X_kg, y, model, n_splits, n_repeats)



# ============================================================
# 11. 执行所有实验
# ============================================================
if __name__ == '__main__':
    results = {}

    # 本文方法：KNN (tuned)
    print("Running KNN (tuned)...")
    results['KNN_tuned'] = run_knn(X_tuned, y)

    # KNN (untuned)
    # print("Running KNN (untuned)...")
    # results['KNN_untuned'] = run_knn(X_untuned, y)

    # # 决策树
    # print("Running Decision Tree...")
    # results['DecisionTree'] = run_decision_tree(X_tuned, y)

    # # K-means
    # print("Running K-means...")
    # results['KMeans'] = run_kmeans(X_tuned, y)

    # TF-IDF + SVM
    print("Running TF-IDF + SVM...")
    results['TFIDF_SVM'] = run_tfidf_svm(texts, y)

    # Word2Vec + SVM
    print("Running Word2Vec + SVM...")
    results['W2V_SVM'] = run_w2v_svm(X_w2v, y)

    # BERT + LR
    print("Running BERT + LR...")
    results['BERT_LR'] = run_bert_lr(X_bert, y)

    # # KG + KNN
    print("Running KG + KNN...")
    results['KG_KNN'] = run_kg_knn(X_kg, y)

    # LLM zero-shot
    llm_preds = llm_zero_shot_predict(texts, call_llm_func)
    results['LLM_zero_shot'] = {'accuracy_mean': accuracy_score(y, llm_preds),
                                'accuracy_std':np.std(accuracy_score(y, llm_preds)),
                                'f1_mean': f1_score(y, llm_preds, average='macro'),
                                'f1_std':np.std(f1_score(y, llm_preds, average='macro')),
                                'weighted_f1_mean': f1_score(y, llm_preds, average='weighted'),
                                'weighted_f1_std':np.std(f1_score(y, llm_preds, average='weighted')),}

    # 汇总结果
    df_results = pd.DataFrame({
        name: {
            'Accuracy': f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}",
            'Macro-F1': f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}",
            'Weighted-F1': f"{res['weighted_f1_mean']:.4f} ± {res['weighted_f1_std']:.4f}",
        }
        for name, res in results.items()
    }).T
    print("\n=== 统一 5 折交叉验证结果 ===")
    print(df_results)

    # ============================================================
    # 12. 统计检验：McNemar 和 Wilcoxon
    # ============================================================
    # 以本文 KNN 为基准，与其他方法比较
    baseline = results['KNN_tuned']
    baseline_accs = baseline['accs']
    baseline_true = baseline['all_y_true']
    baseline_pred = baseline['all_y_pred']

    print("\n=== 统计检验(vs. KNN_tuned) ===")
    for name, res in results.items():
        if name == 'KNN_tuned':
            continue
        # Wilcoxon signed-rank 检验：比较 10 次重复的准确率
        if len(res['accs']) == len(baseline_accs):
            stat, p = wilcoxon(baseline_accs, res['accs'])
            print(f"{name}: Wilcoxon p = {p}")
        # McNemar 检验：基于所有测试折合并的预测
        # 构建 2x2 表
        # 注意：需要对齐样本顺序，这里假设 evaluate_model 返回的 all_y_true 顺序一致
        # 为简化，仅当长度相同时计算
        if len(res['all_y_true']) == len(baseline_true):
            # 正确/错误
            correct_base = (baseline_pred == baseline_true)
            correct_other = (res['all_y_pred'] == res['all_y_true'])
            table = pd.crosstab(correct_base, correct_other)
            # 确保 2x2
            if table.shape == (2, 2):
                result = mcnemar(table.values, exact=False, correction=True)
                print(f"{name}: McNemar p = {result.pvalue}")

    # ============================================================
    # 13. 置信区间（bootstrap 或 t 分布）
    # ============================================================
    print("\n=== 95% 置信区间(Accuracy) ===")
    for name, res in results.items():
        accs = res['accs']
        ci = st.t.interval(0.95, len(accs)-1, loc=np.mean(accs), scale=st.sem(accs))
        print(f"{name}: {np.mean(accs):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    # ============================================================
    # 14. 混淆矩阵（以 KNN_tuned 为例）
    # ============================================================
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(baseline_true, baseline_pred)
    print("\n=== KNN_tuned 混淆矩阵 ===")
    print(cm)
    
    

'''
=== 统一 5 折交叉验证结果 ===
                  Accuracy         Macro-F1      Weighted-F1
KNN_tuned  0.8787 ± 0.0460  0.8418 ± 0.0658  0.8754 ± 0.0486
TFIDF_SVM  0.6563 ± 0.0469  0.4485 ± 0.0862  0.5931 ± 0.0623
W2V_SVM    0.6015 ± 0.0059  0.2517 ± 0.0098  0.4524 ± 0.0092
BERT_LR    0.6467 ± 0.0646  0.5065 ± 0.0838  0.6257 ± 0.0636
KG_KNN     0.9494 ± 0.0310  0.9238 ± 0.0483  0.9476 ± 0.0326


=== 95% 置信区间（Accuracy） ===
KNN_tuned: 0.8787 [0.8655, 0.8919]
TFIDF_SVM: 0.6563 [0.6428, 0.6697]
W2V_SVM: 0.6015 [0.5998, 0.6032]
BERT_LR: 0.6467 [0.6282, 0.6653]
KG_KNN: 0.9494 [0.9405, 0.9583]

=== KNN_tuned 混淆矩阵 ===
[[ 207   72   21]
 [  33 1016   81]
 [  11   10  429]]
'''
