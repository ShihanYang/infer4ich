#  @file: wordset2embedding.py
#  @version：1.0.5
#  @brief: embedding a word set in the .csv file by means of BERT-Chinese-WWM
#          and saving the embedding into a file
#  @creation date: 2025.06.28
#  @last modified date: 2026.4.22 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 

import numpy as np
from transformers import BertTokenizer, BertModel
from pathlib import Path
# import jieba
# import torch
from tqdm import tqdm

# 加载模型和工具
pwd = Path(__file__).resolve().parent  # 当前脚本工作目录

# model_name = pwd / 'model/chinese-wwm-pytorch'   # 基础版
model_name = pwd / 'model/chinese-bert-wwm-ext'   # 哈工大中文bert模型扩展版
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 从文件输入单词集合
wordset = list()
wordset_vect = list()
fileName = pwd / 'data/lisu/relmobility.csv'  # TODO: change csv file to 'Aesthetic'/'History'/'Semiology'/'Sociology'/'bai'/'hani'/'lisu', etc.

with open(fileName, 'r', encoding='UTF-8') as wsf:
    lines = wsf.readlines()
    pbar = tqdm(total = 100)  # 进度条
    for word in lines:
        word = word.strip()  # 去除首尾空格
        word = word.replace(" ", "")  # 去除中间的空格
        word = word.replace("　", "")  # 去除中间的特殊空格"　"
        word = word.replace("（", "")  # 去除中间的特殊符号"（"
        word = word.replace("）", "")  # 去除中间的特殊符号"）"
        wordset.append(word)
        # 获取 BERT 输出
        inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True)
        outputs = model(**inputs)
        char_vectors = outputs.last_hidden_state[0]

        # 对齐分词与 BERT Token 索引
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_indices = []
        current_pos = 1
        for character in word:
            indices = []
            for _ in range(len(character)):
                if current_pos >= len(tokens) or tokens[current_pos] in ["[CLS]", "[SEP]", ".", ",", "、"]:
                    break
                indices.append(current_pos)
                current_pos += 1
            word_indices.append(indices)

        # 池化生成词向量
        word_vectors = []
        for indices in word_indices:
            if not indices:
                continue
            vector = char_vectors[indices]
            word_vectors.append(vector.detach().numpy().flatten())           
             
        # word_vec = np.mean(word_vectors, axis=0)  # 这里是平均池化，也可以Max Pooling捕捉最显著的特征
        word_vec = char_vectors[0].detach().numpy().flatten()   # 这个向量也可以直接是CLS位置对应的向量,CLS位置常常是第一个向量
        
        wordset_vect.append(word_vec)

        # 输出结果
        # print(f"词语: {word}, 向量形状: {word_vec.shape}")
        
        pbar.update(5)
    pbar.close()

# 保存向量到文件 
saveFile = str(fileName) + '.vec'
with open(saveFile, 'w', encoding='UTF-8') as vecf:
    for word, vec in zip(wordset, wordset_vect):
        vecf.write(word + '@' +  str(vec.tolist()) + '\n')  # 因为有向量，向量的分隔符是逗号，所以这里用@分割各个项
        