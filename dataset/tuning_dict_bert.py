#  @file: tuning_dict_bert.py
#  @version：1.0.5
#  @brief: This script is used to fine-tune a BERT model with a user-defined dictionary.
#          Fine-tune the word (term) embedding using its definition text (e.g., the entry in a specialized dictionary).
#  @creation date: 2026.04.24
#  @last modified date: 2026.07.11 
#  @authors: S. Yang
#  @copyright: © 2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv
import os
from transformers import BertTokenizer, BertModel

from pathlib import Path
pwd = Path(__file__).resolve().parent

LOCAL_MODEL_PATH = pwd.parent / 'conditionalProjection' / 'model' / 'chinese-bert-wwm-ext'
USER_DICT_PATH = pwd / 'dictionary'
OUTPUT_DIR = pwd / 'tuned'

# dict_file_name = 'Aesthetic-.csv'   # TODO: change to your dictionary
dict_file_name = 'tuningtest.csv'
dictionary = USER_DICT_PATH / dict_file_name  

def load_local_model():
    if not Path(LOCAL_MODEL_PATH).exists():
        raise FileNotFoundError(f"Model does not exist at: {LOCAL_MODEL_PATH}")
    
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = BertModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    return tokenizer, model

def load_dictionary(data_path):
    """
    加载词典数据
    CSV格式为: term, definition
    --------------------------------------
    'term', 'definition'
    '崇高', '指审美对象在体积、力量或精神境界上超越常规，引起敬畏与赞叹的情感体验。'
    '优美', '指审美对象在形式上和谐、匀称、柔和，引起愉悦和宁静的情感体验。'
    '悲剧', '指主人公遭受不应有的苦难或毁灭，引起怜悯与恐惧，从而净化心灵的艺术形态。'
    '喜剧', '指通过夸张、讽刺等手法表现生活中的矛盾，引起笑声和思考的艺术形态。'
    '意境', '指艺术作品中情景交融、虚实相生，能引发读者无限遐想的审美空间。'
     ...
    -------------------------------------
    """
    train_examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row['term'].strip()
            definition = row['definition'].strip()
            if term and definition:
                train_examples.append((term, definition))
    return train_examples

def mean_pooling(model_output, attention_mask):
    """
    对 BERT 输出进行平均池化
    """
    # model_output 是 BaseModelOutput 对象，提取 last_hidden_state
    token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
    # 扩展 attention_mask 以匹配 embedding 维度
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # 计算加权和
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # 计算非零元素个数 (避免除以0)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask

class TermDefDataset(Dataset):
    """
    自定义数据集，负责将文本转换为 Token IDs
    """
    def __init__(self, examples, tokenizer, max_len=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        term, definition = self.examples[idx]
        
        # 编码 Term
        term_enc = self.tokenizer(
            term, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        
        # 编码 Definition
        def_enc = self.tokenizer(
            definition, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        
        # 返回字典，方便 DataLoader 合并
        return {
            'term_ids': term_enc['input_ids'].squeeze(),      # 去掉 batch 维度 [1, seq] -> [seq]
            'term_mask': term_enc['attention_mask'].squeeze(),
            'def_ids': def_enc['input_ids'].squeeze(),
            'def_mask': def_enc['attention_mask'].squeeze()
        }

def fine_tuning():
    
    # 0. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 设置参数
    BATCH_SIZE = 16
    EPOCHS = 8
    LEARNING_RATE = 1e-5
    MAX_LEN = 128
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = OUTPUT_DIR / dict_file_name.split('.')[0]
    
    # 2. 加载预训练模型
    tokenizer, model = load_local_model()
    
    # 3. 准备数据
    data_file = dictionary
    if not os.path.exists(data_file):  # 如果数据文件不存在，则创建一个示例数据文件
        with open(data_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['term', 'definition'])
            writer.writerow(['崇高', '指审美对象在体积、力量或精神境界上超越常规，引起敬畏与赞叹的情感体验。'])
            writer.writerow(['优美', '指审美对象在形式上和谐、匀称、柔和，引起愉悦和宁静的情感体验。'])
            writer.writerow(['悲剧', '指主人公遭受不应有的苦难或毁灭，引起怜悯与恐惧，从而净化心灵的艺术形态。'])
            writer.writerow(['喜剧', '指通过夸张、讽刺等手法表现生活中的矛盾，引起笑声和思考的艺术形态。'])
            writer.writerow(['意境', '指艺术作品中情景交融、虚实相生，能引发读者无限遐想的审美空间。'])
    train_examples = load_dictionary(data_file)
    if not train_examples:
        raise ValueError("No training data loaded. Please check your CSV file.")
    
    # 4. 创建 Dataset 和 DataLoader
    dataset = TermDefDataset(train_examples, tokenizer, max_len=MAX_LEN)  # 使用自定义 Dataset 进行 Tokenization
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    print(dataloader.dataset)
    
    # 5. 优化器和损失函数    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CosineEmbeddingLoss()
    
    # 6. 训练模型
    print("Tuning...")
    model.to(device)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            # print("  ", type(batch), batch.keys())
            optimizer.zero_grad()
            
            term_out = model(input_ids=batch['term_ids'].to(device), attention_mask=batch['term_mask'].to(device))
            term_vec = mean_pooling(term_out, batch['term_mask'].to(device))
            
            def_out = model(input_ids=batch['def_ids'].to(device), attention_mask=batch['def_mask'].to(device))
            def_vec = mean_pooling(def_out, batch['def_mask'].to(device))
            
            labels = torch.ones(term_vec.size(0)).to(device)
            loss = loss_fn(term_vec, def_vec, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    print("Tuned!")
    
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Tuning completed & Model was saved to {OUTPUT_PATH}.")
    
            
if __name__ == '__main__':
    fine_tuning()
