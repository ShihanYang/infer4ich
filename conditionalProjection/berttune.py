#  @file: berttune.py
#  @version：1.0.5
#  @brief: # Tunning Transformer-based模型
#  @creation date: 2025.08.28
#  @last modified date: 2025.10.12 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 

# Tunning Transformer-based模型（如BERT、RoBERTa等）可以生成上下文相关的词向量。与传统的静态词向量不同，这些模型的词向量会根据上下文动态变化。
# 在具体实践中，微调BERT模型可能需要一定的计算资源和时间，但通常可以为特定任务提供出色的性能。
# 关键challenge：如何准备自己的调优数据！

'''
Bert-base-chinese模型是一个在简体和繁体中文文本上训练得到的预训练模型，具有以下特点：
    12个隐层
    输出768维张量
    12个自注意力头
    110M参数量
    
在运用bert-base-chinese模型时，既能够将其当作特征提取器，把输入文本转化为固定长度的向量表示，随后将这些向量输入至其他机器学习模型中开展训练或推断；
也能够对bert-base-chinese进行微调，使其适配特定任务的训练。
'''

'''
bert-base-chinese微调:
    1. 首先我们需要通过同样的方法来构建基础模型。然后通过语料样本来进行微调。需要自己的数据集！！
    2. 将数据集准备好之后，使用模型的分词器对其进行处理，将处理之后的数据放入模型进行训练。
    3. 训练完之后再测试集上进行预测查看训练效果。并将模型保存。
模型微调只是针对某种下游任务，针对性强化模型的能力，但是微调之后的模型在泛化能力上有所下降。
'''
  
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline   
from datasets import load_dataset, load_from_disk    # ToDo 需要重新准备数据集
import re   
from sklearn.metrics import accuracy_score   

modelpath = r'.\model\bert-base-chinese'   
tokenizer = BertTokenizer.from_pretrained(modelpath)   # 加载分词器
model = BertForSequenceClassification.from_pretrained(modelpath, num_labels=3)   # 加载预训练模型

# 调优前分类器(这里实验一个情感分类器)
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
sentences = ["今天我的心情好到爆了。",
             "用Python编程指南。",
             "我今天太伤心了！"] 
print("Before tunning:")
for s in sentences:
    output = classifier(s)  # 这个分类器的标签是什么？它的分类结果比较随机。
    print("  ", end='')
    print(output)
    
# 获取classifer的标签集合
label_map = classifier.model.config.label2id
labels = list(label_map.keys())
print("标签集合:", labels)  # ['LABEL_0', 'LABEL_1', 'LABEL_2']

# 装载调优数据
# dataset = load_dataset('lansinuote/ChnSentiCorp')   # 加载内置数据集，网络原因常常失败：
                                                      # ConnectionError: Couldn't reach 'lansinuote/ChnSentiCorp' on the Hub (LocalEntryNotFoundError)    

# dataset = load_dataset("csv", data_files={"train": r"F:\mycodes\wordev\data\train.csv"})  # 加载本地数据集合

# 或者这样读文件夹组织的数据
dataset_path = r".\data\lansinuote\ChnSentiCorp"
dataset = load_from_disk(dataset_path)

print(dataset)  # 查看数据集描述
# train_data = dataset["train"]
# print(train_data[0:5])   # 查看数据，数据可以用pandas来处理

# 简单的文本清理函数
def clean_text(text):  
     text = re.sub(r'[^\w\s]+', ' ', text)       
     text = text.strip()       
     return text      

# 预处理数据
dataset = dataset.map(lambda x: {'text': clean_text(x['text']), 'label': x['label']})      

# 形成token函数
def tokenize_function(examples):   
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)      

# 构造token数据集
encoded_dataset = dataset.map(tokenize_function, batched=True)      

# 设置训练参数
training_args = TrainingArguments(  
     output_dir='./results',       
     num_train_epochs=1,       
     per_device_train_batch_size=32,       
     per_device_eval_batch_size=32,      
     eval_strategy='epoch',       
     logging_dir='./logs'  
)      
# 构造训练对象
trainer = Trainer(   
    model=model,       
    args=training_args,       
    train_dataset=encoded_dataset['train'],       
    eval_dataset=encoded_dataset['validation'], 
)  
# 调优训练 
trainer.train()   # 训练很费时！本机一般一个迭代10s，32batches，9600samples，300个it每epoch，时间大约50分钟 
                  # 也费空间，需要大约5G的内存  

# 评估调优后的模型
trainer.evaluate(encoded_dataset['test'],metric_key_prefix='eval')     

# 保存模型 
model.save_pretrained('./sentiment_model')   
tokenizer.save_pretrained('./sentiment_model')   



###############################
# 调优后，使用新模型
###############################

from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline      

mode_dir = './sentiment_model'      

model = AutoModelForSequenceClassification.from_pretrained(mode_dir)   
tokenizer = BertTokenizer.from_pretrained(mode_dir)      

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)  
print("After tunning:")
for s in sentences:
    output = classifier(s)
    print("  ", end='')
    print(output)




#####################################
# 训练一个epoch之后的结果
#####################################
'''
Before tunning:
  [{'label': 'LABEL_2', 'score': 0.5859296917915344}]
  [{'label': 'LABEL_2', 'score': 0.5015413761138916}]
  [{'label': 'LABEL_2', 'score': 0.4971022307872772}]
After tunning:
  [{'label': 'LABEL_1', 'score': 0.6562440395355225}]
  [{'label': 'LABEL_0', 'score': 0.5355194211006165}]
  [{'label': 'LABEL_0', 'score': 0.6313782334327698}]
  
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 9600
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1200
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1200
    })
})  
'''