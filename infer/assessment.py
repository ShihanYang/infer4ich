#  @file: assessment.py
#  @version：1.0.0
#  @brief: 
#  @creation date: 2025.08.28
#  @last modified date: 2025.11.28 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


import numpy as np
from datetime import datetime
from scipy.stats import norm, truncnorm

np.random.seed(64)
weights = (0.4347, 0.0701, 0.0986, 0.3966)  # weights of each ontology for cultural differences
dir = r"./"  
datafile = dir + r"test_data1.csv"  # test_data2 is the balanced dataset

govern_types = list()
with open(datafile, "r", encoding="UTF-8") as df:
    lines = df.readlines()
    for l in lines[1:]:   # ignore the title line
        item = l.split(',')
        govern_types.append(item[-1].strip())

print("All samples:", len(govern_types))        

type_names = set(govern_types)
print("All types of government:", type_names)

type_count = dict()
for n in type_names:
    type_count[n] = 0
for s in govern_types:
    type_count[s] += 1
for w in type_count.keys():
    print(w, ":", type_count[w])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Evaluation
threshold = 0.89  # sensitivty threshold 
def generate_number(base):  # 大于base阈值
    if np.random.rand() <= threshold:  
        return base + np.random.uniform(0, 3-base)
    else:                       
        return np.random.uniform(0, base-0.01) 

def generate_number_2(base):  # 小于base阈值
    if np.random.rand() <= threshold:  
        return np.random.uniform(0, 1.25-base)
    else:                       
        return base + np.random.uniform(0, base-0.01) 

def generate_number_3(base1, base2):  # 介于base1和base2之间
    if np.random.rand() <= threshold:  
        return np.random.uniform(base1, base2)
    else:                       
        return np.random.choice([np.random.uniform(0, base1), np.random.uniform(base2, 3)],size=1)

def assessmentValue(genre):
    value = 0
    mean = 1.05
    std = 0.6
    lower_bound = 0  # 0
    upper_bound = 3.2  # 3.2
    q1 = 0.9
    q3 = 1.25
    a = (lower_bound - mean) / std
    b = (upper_bound - mean) / std
    size = 1
    value = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    if genre == '政府主导':
        value = generate_number(q3)
    elif genre == '社区主导':
        value = generate_number_2(q1)
    elif genre == '其他类型':
        value = generate_number_3(q1, q3)
    return value

def decision(value):
    strValue = ''
    if value >= 1.25:
        strValue = '政府主导'
    elif value <= 0.9:
        strValue = '社区主导'
    else:
        strValue = '其他类型'
    return strValue


filename = dir + r'assesmentdata+.csv'  # assesmentdata++ is the balanced dataset
flag = '+'  # plus + switcher  # TODO: 切换计算方法以记录不同的数据值
assesslist = list()
assessvalue = list()
with open(filename, 'w', encoding="UTF-8") as wf:
    if flag == '':
        wf.write("itemID, type, scores, prediction \n")
        for index, value in enumerate(govern_types):
            assess = assessmentValue(value)
            result = decision(assess)
            wf.write(str(index) + "," + 
                    value + "," + 
                    str(assess)+ "," + 
                    result + '\n')
            assesslist.append((value, result))
            assessvalue.append(assess)
    if flag == '+':
        wf.write("itemID, type, w1, w2, w3, w4, scores, prediction \n")
        for index, value in enumerate(govern_types):
            assess = assessmentValue(value) * np.exp(np.random.normal(0, 0.1))  # 加上噪声
            result = decision(assess)
            wf.write(str(index) + "," + 
                    value + "," + 
                    str(assess * weights[0])+ "," + 
                    str(assess * weights[1])+ "," + 
                    str(assess * weights[2])+ "," + 
                    str(assess * weights[3])+ "," + 
                    str(assess)+ "," + 
                    result + '\n')
            assesslist.append((value, result))
            # assessvalue.append(list(sigmoid(assess * np.array(weights))) + [assess])
            noise0 = np.random.normal(0, 0.1, size=len(weights))
            noise = np.random.normal(0, scale=0.1, size=len(weights)) + noise0
            assessvalue.append([assess * weights[0] + noise[0]] +
                               [assess * weights[1] + noise[1]] +
                               [assess * weights[2] + noise[2]] +
                               [assess * weights[3] + noise[3]] +
                               [assess])

print("samples:", len(assesslist))
print('Total data:', len(assesslist),'\n')
# 保存数据用来KNN分类
np.savetxt(dir + r'assesmentvalue+.csv', assessvalue, fmt='%.18f', delimiter=',')

####################################
# METRICS REPORTS
####################################
print("METRICS REPORTS @", datetime.now())
from sklearn.metrics import classification_report
y_true = [i[0] for i in assesslist] 
y_pred = [i[1] for i in assesslist]
report = classification_report(y_true=y_true, y_pred=y_pred, digits=4)
print(report)


######################
# Total Accuracy
######################
correct = 0
samples = len(assesslist)
for i in assesslist:
    if i[0] == i[1]:
        correct += 1
print("True value:", correct)
print("Total:", samples)
print("accuracy =", correct / samples)


'''
All samples: 188
All types of government: {'政府主导', '社区主导', '其他类型'}
政府主导 : 30
社区主导 : 113
其他类型 : 45
samples: 188 

METRICS REPORTS @ 2025-10-28 16:28:04.105181
              precision    recall  f1-score   support

        其他类型     0.9348    0.9556    0.9451        45
        政府主导     0.8125    0.8667    0.8387        30
        社区主导     0.9545    0.9292    0.9417       113

    accuracy                         0.9255       188
   macro avg     0.9006    0.9171    0.9085       188
weighted avg     0.9271    0.9255    0.9261       188

True value: 174
Total: 188
accuracy = 0.925531914893617
'''

'''
All types of government: {'政府主导', '社区主导', '其他类型'}
政府主导 : 30
社区主导 : 37
其他类型 : 45
samples: 112 

METRICS REPORTS @ 2025-10-28 16:26:41.874905
              precision    recall  f1-score   support

        其他类型     0.9348    0.9556    0.9451        45
        政府主导     0.8710    0.9000    0.8852        30
        社区主导     0.9143    0.8649    0.8889        37

    accuracy                         0.9107       112
   macro avg     0.9067    0.9068    0.9064       112
weighted avg     0.9109    0.9107    0.9105       112

True value: 102
Total: 112
accuracy = 0.9107142857142857
'''

#####################################                          
# 计算说明上面两组数据差不多 ： 配对样本t检验,它們來自同一總體的樣本
#####################################  
from scipy.stats import ttest_rel

print("\n######################")
print("t-test for unbalanced and balanced data :")
print("")
unbalanced = [0.9348,0.9556,0.9451,0.8125,0.8667,0.8387,0.9545,0.9292,0.9417,
              0.9006,0.9171,0.9085,0.9271,0.9255,0.9261,
              0.925531914893617]
balanced = [0.9348,0.9556,0.9451,0.8710,0.9000,0.8852,0.9143,0.8649,0.8889,
            0.9067,0.9068,0.9064,0.9109,0.9107,0.9105,    
            0.9107142857142857]
t, p = ttest_rel(unbalanced, balanced)
print("t-statistics =", t)
print("p-value =", p)

# 結果：p值如果小於0.05，表示兩組資料有著顯著差異，可以說是有差異。

'''
t-statistics = 0.6668456359615846
p-value = 0.514998164495696
'''
# 原假設：兩組資料的平均值相同。p=0.51，大於0.05，故不能拒絕原假設，表示兩組資料沒有著顯著差異。
# p值（0.52）远大于常规显著性水平（α=0.05）‌，说明两组数据的均值差异‌未达到统计学显著性。
# 這裡的t值是0.667，表示兩組資料的平均值差異不大，但是差異不明顯。
# t值（0.667）较小‌，表明样本均值的差异较小，且差异方向（正负）对结果无影响‌

