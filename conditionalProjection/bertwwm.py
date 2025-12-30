from transformers import BertTokenizer, BertModel
import jieba
import torch

# 加载模型和工具
# model_name = r'F:\mycodes\wordev\model\chinese-wwm-pytorch'   # 基础版
model_name = r'F:\mycodes\wordev\model\chinese-bert-wwm-ext'   # 哈工大中文bert模型扩展版
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入句子
sentence = "我喜欢吃富士苹果，这种苹果它甜得很，你不吃就算了，Yeah！"
words = jieba.lcut(sentence)
print(words)   # 显然这里要去除一些tokens，比如句号和助词等等 

# 获取 BERT 输出
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
outputs = model(**inputs)
char_vectors = outputs.last_hidden_state[0]

# 对齐分词与 BERT Token 索引
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
word_indices = []
current_pos = 1
for word in words:
    indices = []
    for _ in range(len(word)):
        if current_pos >= len(tokens) or tokens[current_pos] in ["[CLS]", "[SEP]"]:
            break
        indices.append(current_pos)
        current_pos += 1
    word_indices.append(indices)

# 池化生成词向量
word_vectors = []
for indices in word_indices:
    if not indices:
        word_vectors.append(None)
        continue
    vectors = char_vectors[indices]
    word_vec = torch.mean(vectors, dim=0)  # 这里是求平均，也可以试试max函数
    word_vectors.append(word_vec.detach().numpy())

# 输出结果
for word, vec in zip(words, word_vectors):
    print(f"词语: {word}, 向量形状: {vec.shape if vec is not None else 'None'}")