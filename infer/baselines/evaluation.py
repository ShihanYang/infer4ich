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
    confusion_matrix
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

# ============================================================
# 1. Data Ready
# ============================================================
# 假设：
# y: 形状 (n_samples,)，标签 0=government-led, 1=community-led, 2=mixed
# X_tuned: 本文方法微调后的特征向量 (n_samples, n_features)
# X_untuned: 未微调特征向量
# texts: 原始文本列表，长度 n_samples
# X_w2v: Word2Vec/GloVe 平均向量 (n_samples, dim)
# X_bert: 未微调 BERT [CLS] 向量 (n_samples, 768)
# X_kg: TransE/R-GCN 图嵌入 (n_samples, dim)
# 另外需要 LLM zero-shot 的预测结果 llm_preds（长度 n_samples，或每个测试折单独调用）

# 示例：随机生成假数据，仅用于跑通流程
np.random.seed(42)
n_samples = 188
n_features = 20
X_tuned = np.random.randn(n_samples, n_features)
X_untuned = np.random.randn(n_samples, n_features)
y = np.array([0]*30 + [1]*113 + [2]*45)
np.random.shuffle(y)
texts = ["这是一个非物质文化遗产项目的描述。" for _ in range(n_samples)]
X_w2v = np.random.randn(n_samples, 200)
X_bert = np.random.randn(n_samples, 768)
X_kg = np.random.randn(n_samples, 64)

# 平衡数据集索引：下采样 community-led (label=1) 到 37
idx_gov = np.where(y == 0)[0]
idx_com = np.where(y == 1)[0]
idx_mix = np.where(y == 2)[0]
np.random.shuffle(idx_com)
idx_com_balanced = idx_com[:37]
idx_balanced = np.concatenate([idx_gov, idx_com_balanced, idx_mix])
np.random.shuffle(idx_balanced)

# ============================================================
# 2. 统一评估函数
# ============================================================
def evaluate_model(X, y, model, n_splits=5, n_repeats=10, random_state=42):
    """
    使用 RepeatedStratifiedKFold 评估模型。
    X: 特征矩阵或文本列表（如果模型是 Pipeline 支持文本）
    y: 标签
    model: 实现 fit/predict 的分类器
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
def run_knn(X, y, k=7, n_splits=5, n_repeats=10):
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    return evaluate_model(X, y, model, n_splits, n_repeats)

# ============================================================
# 4. 决策树
# ============================================================
def run_decision_tree(X, y, n_splits=5, n_repeats=10):
    model = DecisionTreeClassifier(random_state=42)
    return evaluate_model(X, y, model, n_splits, n_repeats)

# ============================================================
# 5. K-means 分类器（K-means + 多数投票）
# ============================================================
class KMeansClassifier:
    def __init__(self, n_clusters=3, random_state=42):
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
    model = KMeansClassifier(n_clusters=3, random_state=42)
    return evaluate_model(X, y, model, n_splits, n_repeats)

# ============================================================
# 6. TF-IDF + SVM
# ============================================================
def jieba_tokenizer(text):
    return list(jieba.cut(text))

def run_tfidf_svm(texts, y, n_splits=5, n_repeats=10):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=5000, ngram_range=(1,2))),
        ('svm', LinearSVC(random_state=42, dual='auto'))
    ])
    # 注意：evaluate_model 需要支持 list 输入
    return evaluate_model(texts, y, pipeline, n_splits, n_repeats)

# ============================================================
# 7. Word2Vec / GloVe + SVM
# ============================================================
# 假设已经加载了预训练词向量模型 w2v_model，支持 model[word] -> vector
# 如果没有，可以用 gensim 加载：
# from gensim.models import KeyedVectors
# w2v_model = KeyedVectors.load_word2vec_format('path/to/vectors.bin', binary=True)

def doc_vector(text, model, dim=200):
    words = jieba.cut(text)
    vecs = [model[w] for w in words if w in model]
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

def build_w2v_features(texts, model, dim=200):
    return np.array([doc_vector(t, model, dim) for t in texts])

def run_w2v_svm(X_w2v, y, n_splits=5, n_repeats=10):
    model = SVC(kernel='rbf', random_state=42)
    return evaluate_model(X_w2v, y, model, n_splits, n_repeats)

# ============================================================
# 8. 未微调 BERT + Logistic Regression
# ============================================================
# 假设已经提取好 X_bert 特征（n_samples, 768）
def run_bert_lr(X_bert, y, n_splits=5, n_repeats=10):
    model = LogisticRegression(max_iter=1000, random_state=42)
    return evaluate_model(X_bert, y, model, n_splits, n_repeats)

# 如果需要从头提取 BERT 特征，可以用以下代码：
def extract_bert_features(texts, model_name='hfl/chinese-bert-wwm-ext', max_length=512):
    from transformers import BertTokenizer, BertModel
    import torch
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    bert.eval()
    features = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        features.append(cls_embedding)
    return np.array(features)

# ============================================================
# 9. LLM zero-shot 分类
# ============================================================
# 由于需要调用外部 API，这里给出函数框架。
# 你需要实现 call_llm(prompt) 函数，返回模型输出文本。

def llm_zero_shot_predict(texts, call_llm_func):
    """
    对每个文本调用 LLM，返回预测标签数组。
    假设 LLM 输出 'A', 'B', 'C'，分别映射到 0, 1, 2。
    """
    label_map = {'A': 0, 'B': 1, 'C': 2}
    preds = []
    for text in texts:
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
    return np.array(preds)

# 使用示例：
# def call_llm(prompt):
#     # 调用 DeepSeek/GPT API
#     return "A"
# llm_preds = llm_zero_shot_predict(texts, call_llm)
# 由于 zero-shot 无训练，可以直接计算准确率：
# acc = accuracy_score(y, llm_preds)

# ============================================================
# 10. TransE / R-GCN 图嵌入 + KNN
# ============================================================
# 假设已经用 pykeen 或 dgl 训练好图嵌入，得到 X_kg (n_samples, dim)
# 这里给出使用 pykeen 的简化示例，实际需要构建三元组。

def run_kg_knn(X_kg, y, n_splits=5, n_repeats=10):
    model = KNeighborsClassifier(n_neighbors=7)
    return evaluate_model(X_kg, y, model, n_splits, n_repeats)

# 如果需要从头训练 TransE：
# from pykeen.pipeline import pipeline
# result = pipeline(
#     model='TransE',
#     training=training_triples,
#     testing=testing_triples,
#     validation=validation_triples,
#     model_kwargs=dict(embedding_dim=64),
#     training_kwargs=dict(num_epochs=100),
# )
# entity_embeddings = result.model.entity_representations[0]().detach().numpy()
# 然后根据项目节点索引取对应嵌入。

# ============================================================
# 11. 运行所有实验
# ============================================================
if __name__ == '__main__':
    results = {}

    # 本文方法：KNN (tuned)
    print("Running KNN (tuned)...")
    results['KNN_tuned'] = run_knn(X_tuned, y)

    # KNN (untuned)
    print("Running KNN (untuned)...")
    results['KNN_untuned'] = run_knn(X_untuned, y)

    # 决策树
    print("Running Decision Tree...")
    results['DecisionTree'] = run_decision_tree(X_tuned, y)

    # K-means
    print("Running K-means...")
    results['KMeans'] = run_kmeans(X_tuned, y)

    # TF-IDF + SVM
    print("Running TF-IDF + SVM...")
    results['TFIDF_SVM'] = run_tfidf_svm(texts, y)

    # Word2Vec + SVM（假设 w2v_model 已加载）
    # print("Running Word2Vec + SVM...")
    # results['W2V_SVM'] = run_w2v_svm(X_w2v, y)

    # BERT + LR
    print("Running BERT + LR...")
    results['BERT_LR'] = run_bert_lr(X_bert, y)

    # KG + KNN
    print("Running KG + KNN...")
    results['KG_KNN'] = run_kg_knn(X_kg, y)

    # LLM zero-shot（需要 API）
    # llm_preds = llm_zero_shot_predict(texts, call_llm)
    # results['LLM_zero_shot'] = {'accuracy_mean': accuracy_score(y, llm_preds)}

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

    print("\n=== 统计检验（vs. KNN_tuned） ===")
    for name, res in results.items():
        if name == 'KNN_tuned':
            continue
        # Wilcoxon signed-rank 检验：比较 10 次重复的准确率
        if len(res['accs']) == len(baseline_accs):
            stat, p = wilcoxon(baseline_accs, res['accs'])
            print(f"{name}: Wilcoxon p = {p:.4f}")
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
                print(f"{name}: McNemar p = {result.pvalue:.4f}")

    # ============================================================
    # 13. 置信区间（bootstrap 或 t 分布）
    # ============================================================
    print("\n=== 95% 置信区间（Accuracy） ===")
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