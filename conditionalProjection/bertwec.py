################
# 使用BERT模型
################

from transformers import BertTokenizer, BertModel

#加载BERT的分词器和模型 
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = 
                                        r'F:\mycodes\wordev\model\bert-base-chinese')  
# 如果没有路径会自动下载模型from the following websites，但是速度是个问题
model = BertModel.from_pretrained(pretrained_model_name_or_path = 
                                r'F:\mycodes\wordev\model\bert-base-chinese')  
# "https://huggingface.co/google-bert/bert-base-chinese" 
# "https://huggingface.co/models"

#输入句子并分词 
sentence = "我爱中国"  # ，我来自中国云南省昆明市"
tokens = tokenizer.tokenize(sentence)
print(tokens)  # ['我', '爱', '中', '国']
inputs = tokenizer(sentence, return_tensors='pt')
print(inputs)  # {'input_ids': tensor([[ 101, 2769, 4263,  704, 1744,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
print(inputs['input_ids'])

decoded_text = tokenizer.decode(inputs['input_ids'][0])  # [CLS] 我 爱 中 国 [SEP]
print(decoded_text)  # 显然这个分词结果并不好，这个是分字不是分词 ？？ 试试哈工大的模型：Chinese-BERT-wwm 

#获取隐藏层输出 
outputs = model(**inputs)

#提取最后一层隐藏状态作为embeddingvector 
last_hidden_states = outputs.last_hidden_state

#打印第一个词的embeddingvector (CLS token)
vct = last_hidden_states[0][0]
print(len(vct), type(vct))
print(last_hidden_states.shape)
