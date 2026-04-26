#  @file: tuning_dict_bert.py
#  @version：1.0.0
#  @brief: This script is used to test the fine-tuned BERT model enhanced by (term, definition) pairs.
#  @creation date: 2026.04.24
#  @last modified date: 2026.04.25 
#  @authors: S. Yang
#  @copyright: © 2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from pathlib import Path

pwd = Path(__file__).resolve().parent
TUNED_MODEL_PATH = pwd / 'tuned' / 'Aesthetic-'  # TODO: Change this to the path of your tuned model
# 如果未微调，可切换回原始模型路径进行对比
# ORIGINAL_MODEL_PATH = pwd.parent / 'conditionalProjection' / 'model' / 'chinese-bert-wwm-ext'
# TUNED_MODEL_PATH = ORIGINAL_MODEL_PATH

def load_tuned_model():
    """加载微调后的模型和分词器"""
    if not TUNED_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到微调后的模型路径: {TUNED_MODEL_PATH}")
    
    print(f"正在加载模型从: {TUNED_MODEL_PATH}")
    tokenizer = BertTokenizer.from_pretrained(TUNED_MODEL_PATH)
    model = BertModel.from_pretrained(TUNED_MODEL_PATH)
    
    # 设置为评估模式（关闭 Dropout 等）
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model, device

def get_embedding(text, tokenizer, model, device, max_len=128):
    """
    将文本转换为固定维度的向量 (Mean Pooling)
    """
    inputs = tokenizer(
        text, 
        padding='max_length', 
        truncation=True, 
        max_length=max_len, 
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # Mean Pooling 逻辑 (与训练时保持一致)
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    return embedding.squeeze(0) # 返回 [hidden_size]

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return F.cosine_similarity(vec1, vec2, dim=0).item()

def main():
    # 1. 加载模型
    tokenizer, model, device = load_tuned_model()
    
    # 2. 准备测试数据 (术语, 正确定义, 错误定义)
    test_cases = [
        {
            "term": "喜剧",
            "correct_def": "指通过夸张、讽刺等手法表现生活中的矛盾，引起笑声和思考的艺术形态。",  
            "wrong_def": "指艺术作品中情景交融、虚实相生，能引发读者无限遐想的审美空间。" 
        },
        {
            "term": "悲剧",
            "correct_def": "指主人公遭受不应有的苦难或毁灭，引起怜悯与恐惧，从而净化心灵的艺术形态。",
            "wrong_def": "指审美对象在形式上和谐、匀称、柔和，引起愉悦和宁静的情感体验。" 
        }
    ]
    
    print("-" * 50)
    print("开始测试模型语义匹配能力...")
    print("-" * 50)
    
    for i, case in enumerate(test_cases):
        term_vec = get_embedding(case["term"], tokenizer, model, device)
        correct_vec = get_embedding(case["correct_def"], tokenizer, model, device)
        wrong_vec = get_embedding(case["wrong_def"], tokenizer, model, device)
        
        # 计算相似度
        sim_correct = cosine_similarity(term_vec, correct_vec)
        sim_wrong = cosine_similarity(term_vec, wrong_vec)
        
        print(f"\n[测试案例 {i+1}] 术语: 【{case['term']}】")
        print(f"  与【正确定义】的相似度: {sim_correct:.4f}")
        print(f"  与【错误定义】的相似度: {sim_wrong:.4f}")
        
        # 简单判断：如果正确定义的相似度显著高于错误定义，说明微调有效
        if sim_correct > sim_wrong:
            print("  结果: 模型成功区分了正确与错误定义！")
        else:
            print("  结果: 模型未能正确区分，可能需要更多训练或调整超参数。")

    print("-" * 50)

if __name__ == '__main__':
    main()
    
    
'''
正在加载模型从: \tuned\Aesthetic-
--------------------------------------------------
开始测试模型语义匹配能力...
--------------------------------------------------

[测试案例 1] 术语: 【喜剧】
  与【正确定义】的相似度: 0.9238
  与【错误定义】的相似度: 0.8716
  结果: 模型成功区分了正确与错误定义！

[测试案例 2] 术语: 【悲剧】
  与【正确定义】的相似度: 0.9359
  与【错误定义】的相似度: 0.8813
  结果: 模型成功区分了正确与错误定义！
--------------------------------------------------
'''
##################################################
'''
正在加载模型从: \model\chinese-bert-wwm-ext
--------------------------------------------------
开始测试模型语义匹配能力...
--------------------------------------------------

[测试案例 1] 术语: 【喜剧】
  与【正确定义】的相似度: 0.6399
  与【错误定义】的相似度: 0.5650
  结果: 模型成功区分了正确与错误定义！

[测试案例 2] 术语: 【悲剧】
  与【正确定义】的相似度: 0.6190
  与【错误定义】的相似度: 0.5554
  结果: 模型成功区分了正确与错误定义！
--------------------------------------------------
'''
###################################
## 结果分析：
###################################
'''
1.  得分普遍提高说明模型‌学会了关联‌，这一现象揭示了模型在‌语义对齐能力‌
和‌向量空间分布‌上的显著变化，它将一个通用的语义编码器转化为了一个高精
度的领域专用匹配模型，说明‌微调是非常成功且必要的‌。在我们的实验中，这
是足够的。
2.  但区分度低说明模型‌没学会鉴别‌。为了提高区分度（即拉大正负样本得分
差值），可以尝试以下改进：
   1) 引入硬负样本
   2) 调整损失函数
   3) 增加负样本数量
   4) 数据增强
'''
