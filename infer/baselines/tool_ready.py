#  @file: tool_ready.py
#  @version：1.0.0
#  @brief: Ready feature vectors for baseline testing cases
#  @creation date: 2026.10.13
#  @last modified date: 2026.10.15 
#  @authors: S. Yang
#  @copyright: ©2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 



# ============================================================
# Word2vec（/Glove） - embedding X_w2v
# ============================================================
import numpy as np
import jieba
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# 全局配置与模型加载
MODEL_PATH = "tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"   # 需要镜像，原始的门户已经关闭
BIN_MODEL_PATH = r"light_Tencent_AILab_ChineseEmbedding.bin"  # https://modelscope.cn/models/lili666/text2vec-word2vec-tencent-chinese/files

def load_w2v_model():
    """加载预训练 Word2Vec 模型"""
    if os.path.exists(BIN_MODEL_PATH):
        return KeyedVectors.load_word2vec_format(BIN_MODEL_PATH, binary=True)
    elif os.path.exists(MODEL_PATH):
        # 首次加载较慢，加载后自动保存为二进制格式
        model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
        model.save_word2vec_format(BIN_MODEL_PATH, binary=True)
        return model
    else:
        raise FileNotFoundError("未找到预训练模型文件，请先下载腾讯AI Lab词向量。")

# 初始化全局模型
w2v_model = load_w2v_model()
VECTOR_DIM = w2v_model.vector_size

class DocVectorizer:
    def __init__(self, w2v_model, max_features=50000):
        self.w2v_model = w2v_model
        self.vector_dim = w2v_model.vector_size
        # 初始化 TF-IDF 向量器，使用 jieba 分词
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=jieba.lcut,
            token_pattern=None,  # 允许单字作为 token
            max_features=max_features
        )
        self.is_fitted = False

    def fit(self, corpus):
        """
        在语料库上拟合 TF-IDF 模型，学习 IDF 权重。
        :param corpus: list of str, 用于统计词频的文档列表
        """
        self.tfidf_vectorizer.fit(corpus)
        self.is_fitted = True

    def transform(self, text):
        """
        输入单个文档字符串，返回归一化的文档向量。
        :param text: str, 输入文档
        :return: np.array, 固定维度的文档向量
        """
        if not self.is_fitted:
            raise Exception("请先调用 fit() 方法拟合 TF-IDF 模型")
        
        # 1. 获取文本的 TF-IDF 稀疏表示
        tfidf_matrix = self.tfidf_vectorizer.transform([text])
        
        # 2. 提取非零项的索引和权重
        indices = tfidf_matrix.nonzero()[1]
        values = tfidf_matrix.data
        
        if len(indices) == 0:
            return np.zeros(self.vector_dim)
            
        # 3. 获取对应的词语
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # 4. 加权累加 Word2Vec 向量
        weighted_sum = np.zeros(self.vector_dim)
        total_weight = 0
        
        for idx, tfidf_val in zip(indices, values):
            word = feature_names[idx]
            # 仅处理存在于 Word2Vec 词典中的词
            if word in self.w2v_model:
                vec = self.w2v_model[word]
                weighted_sum += tfidf_val * vec
                total_weight += tfidf_val
        
        # 5. 归一化处理
        if total_weight > 0:
            final_vec = weighted_sum / total_weight
        else:
            # 如果所有有效词的向量都缺失，返回零向量
            return np.zeros(self.vector_dim)
            
        # L2 归一化，便于后续计算余弦相似度
        norm = np.linalg.norm(final_vec)
        if norm > 0:
            return final_vec / norm
        return final_vec

# Data Accessing API
def text_to_vector(corpus, text, model=w2v_model, vector_size=128):
    # 1. 准备一批业务语料用于拟合 TF-IDF (IDF 统计需要全局视角)
    train_corpus = corpus
    # 2. 初始化并向量化器
    vectorizer = DocVectorizer(model)
    # 3. 拟合 TF-IDF (只需执行一次)
    vectorizer.fit(train_corpus)
    textvec = vectorizer.transform(text)
    return textvec



# ============================================================
# BERT - embedding X_bert
# ============================================================
bert_model_name = r"chinese-bert-wwm-ext"
# Data Accessing API
def extract_bert_features(texts, model_name=bert_model_name, max_length=512):
    from transformers.models.bert import BertTokenizer, BertModel
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
# TransE (/R-GCN) - embedding X_kg
# ============================================================
# PyKEEN 是知识图谱嵌入（KGE，如 TransE/RotatE/ComplEX）的专用工具；
# DGL 是通用图神经网络框架，灵活度高，适合训练 R-GCN/GAT 等融合拓扑结构的深度图嵌入‌。
# Data Accessing API
def get_kg_embedding(path):
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    import torch
    # 加载自定义三元组
    tf = TriplesFactory.from_path(
        path=path,  
        delimiter="\t",
        encoding="utf-8",
        create_inverse_triples=True,  # 自动添加反向关系，提升效果
    )
    # print(f"实体数: {tf.num_entities}, 关系数: {tf.num_relations}, 三元组数: {tf.num_triples}")  # 存在相同的实体和关系，因此数量少于总三元组数
    # 划分训练/验证/测试集
    training, validation, testing = tf.split([0.8, 0.1, 0.1],  method='cleanup', random_state=68)
    # 训练模型，训练代价比较大
    result = pipeline(
        model='TransE',
        training=training,
        testing=testing,
        validation=validation,
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=100, batch_size=128),
        random_seed=68,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # 获取模型
    model = result.model
    # 获取嵌入向量
    entity_embeddings = model.entity_representations[0]().detach().numpy()
    relation_embeddings = model.relation_representations[0]().detach().numpy()
    # 获取实体到ID的映射字典：实体名 -> 嵌入索引
    entity_to_id = training.entity_to_id
    
    # # 保存嵌入到本地
    # np.save("entity_embeddings_64d.npy", entity_embeddings)
    # np.save("relation_embeddings_64d.npy", relation_embeddings)
    # # 保存实体-id映射供后续使用
    # import json
    # with open("entity_to_id.json", "w", encoding="utf-8") as f:
    #     json.dump(entity_to_id, f, ensure_ascii=False, indent=2)
    # # 保存测试集评估结果
    # result.metric_results.to_df().to_csv("kge_metrics.csv", index=False)
    # # 保存完整模型
    # result.save_to_directory("./transe_model")
        
    return entity_embeddings, relation_embeddings, entity_to_id



# ============================================================
# LLM Zero Shot - list of answers
# ============================================================
# Data Accessing API
def call_llm_func(prompt):
    from openai import OpenAI
    client = OpenAI(
        api_key="sk-*********************",  
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": "你是一个非物质文化遗产保护专家。"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    return response.choices[0].message.content




# ============================================================
# testing anytime
# ============================================================
if __name__ == "__main__":
    text = """一得阁墨汁制作技艺是北京市级非物质文化遗产，位于北京市东城区东华门内大街，由一得阁墨汁制作技艺传承人王文杰传授。"""
    prompt= f"""你是一个非物质文化遗产保护专家。请阅读以下项目描述，判断最适合的保护策略类型。
                类别：
                A. 政府主导
                B. 社区主导
                C. 混合型
                只输出 A、B 或 C。

                项目描述：
                {text}
    """
    answer = call_llm_func(prompt)
    print(answer)


if __name__ == "__main_1__":
    triples_file = r"triples.tsv"
    entity_embeddings, _, entity_to_id = get_kg_embedding(triples_file)
    ai_vec = entity_embeddings[entity_to_id["一得阁墨汁制作技艺"]]
    print(f"实体'一得阁墨汁制作技艺'的嵌入维度：{ai_vec.shape}")  
    ai_vec = entity_embeddings[entity_to_id["社区主导"]]
    print(f"实体'社区主导'的嵌入：{ai_vec}")  
    '''
    实体嵌入维度：(160, 64)
    实体'一得阁墨汁制作技艺'的嵌入维度：(64,)
    实体'社区主导'的嵌入：[-0.06302773 -0.27624068  0.04365262 -0.07242507 -0.0446507  -0.05289925
    -0.11347024  0.0539151  -0.17966262 -0.2965137  -0.2988623  -0.04490935
    0.21532932 -0.08621106 -0.05301237 -0.03883236 -0.2786289  -0.06810202
    0.10185596  0.03884588  0.06003215  0.01495646 -0.07219446 -0.30814454
    0.04349933 -0.03863462 -0.10067265 -0.08528085 -0.26518694 -0.31993693
    -0.00741658  0.03115278 -0.00810865  0.21558629  0.07003235  0.05557468
    -0.05584066 -0.0640847  -0.00867042 -0.03576542  0.05546473  0.05205132
    0.0503399   0.02055886  0.19437699  0.00326639 -0.03982452 -0.04785711
    0.09629368  0.05885687  0.10733733 -0.0311647   0.14812331  0.00256329
    0.05224539 -0.02535741  0.02484829  0.06552445 -0.01866734  0.0661688
    -0.04947816 -0.03557214 -0.07549916 -0.21578647]
    '''
    
    
    # -------------------------------------------------------
    # 加载模型并得分估计的推理
    # -------------------------------------------------------
    
    from pykeen.pipeline import PipelineResult
    import torch

    # 1. 从目录加载完整的结果对象
    # 这会自动加载模型、训练三元组工厂（包含实体/关系映射）和元数据
    pipeline_result = PipelineResult.from_directory("./transe_model")

    # 2. 获取模型实例
    model = pipeline_result.model

    # 3. 获取训练时的三元组工厂（用于实体/关系名称到ID的转换）
    training_triples_factory = pipeline_result.training

    # 4. 将模型设置为评估模式（重要：关闭 Dropout 等训练特定行为）
    model.eval()
    
    # 5. 推理
    # 定义要预测的三元组（使用字符串标签）
    head_entity = "一得阁墨汁制作技艺"
    relation = "同治四年，是我"
    tail_entity = "其他类型"

    # 获取实体和关系对应的 ID
    # 注意：如果实体或关系不在训练集中，这里会抛出 KeyError
    try:
        head_id = training_triples_factory.entity_to_id[head_entity]
        rel_id = training_triples_factory.relation_to_id[relation]
        tail_id = training_triples_factory.entity_to_id[tail_entity]
    except KeyError as e:
        print(f"错误: 实体或关系 '{e}' 未在训练集中出现，无法直接预测。")
        head_id, rel_id, tail_id = None, None, None

    if head_id is not None:
        # 构建输入张量
        # score_hrt 期望输入形状为 (batch_size, 3) 或分别传入 h, r, t
        h_tensor = torch.tensor([head_id])
        r_tensor = torch.tensor([rel_id])
        t_tensor = torch.tensor([tail_id])

        # 计算评分 (建议在 no_grad 环境下进行以节省内存)
        with torch.no_grad():
            scores = model.score_hrt(h=h_tensor, r=r_tensor, t=t_tensor)
        
        print(f"三元组 ({head_entity}, {relation}, {tail_entity}) 的评分: {scores.item()}")

    # -------------------------------------------------------
    # 从头实体和关系实体预测尾实体
    # -------------------------------------------------------
    import torch
    from pykeen.pipeline import PipelineResult

    # 1. 加载保存的模型和训练数据上下文
    # 必须使用 from_directory 加载，以确保实体ID映射一致
    pipeline_result = PipelineResult.from_directory("./transe_model")
    model = pipeline_result.model
    training_tf = pipeline_result.training

    # 2. 准备输入：将字符串标签转换为 ID
    head_entity_label = "一得阁墨汁制作技艺"
    relation_label = "同治四年，是我"  # 关系可能不止一个

    # 获取对应的整数 ID
    head_id = training_tf.entity_to_id[head_entity_label]
    rel_id = training_tf.relation_to_id[relation_label]

    # 转换为 Tensor (PyKEEN 期望输入为 LongTensor)
    h_tensor = torch.tensor([head_id])
    r_tensor = torch.tensor([rel_id])

    # 3. 执行推理：预测尾实体
    # model.predict_t 会自动计算 h, r 与所有可能 t 的评分并排序
    # k 表示返回前 k 个最可能的实体
    k = 5
    predicted_t_ids, predicted_scores = model.predict_t(h=h_tensor, r=r_tensor, k=k)

    # 4. 解析结果：将 ID 转换回字符串标签
    predicted_entities = []
    for t_id, score in zip(predicted_t_ids[0], predicted_scores[0]):
        # 注意：predict_t 返回的是 ID，需要通过 factory 映射回标签
        entity_label = training_tf.id_to_entity[t_id.item()]
        predicted_entities.append({
            "entity": entity_label,
            "score": score.item()
        })

    # 5. 输出结果
    print(f"给定 ({head_entity_label}, {relation_label}), 最可能的尾实体为:")
    for item in predicted_entities:
        print(f"- {item['entity']}: {item['score']:.4f}")
