# Evaluate the precision, recall, and F1-score of predicted values relative to true values.

from sklearn.metrics import classification_report


y_true = [1,0,0,0,1,1,1,1,1,1,1,0,0,2,2,2,2,2,2,2,0,0,1]  # 真实标签
y_pred = [0,0,0,0,1,1,1,1,1,1,1,1,0,2,2,2,2,2,2,2,0,1,1]  # 预测标签
report = classification_report(y_true, y_pred)

print(report)


#######################
# report
#######################

'''
              precision    recall  f1-score   support

           0       0.83      0.71      0.77         7
           1       0.80      0.89      0.84         9
           2       1.00      1.00      1.00         7

    accuracy                           0.87        23
   macro avg       0.88      0.87      0.87        23
weighted avg       0.87      0.87      0.87        23
'''

'''
评估分类模型的预测性能，基于预测值（predicted values）与真实值（true values）的对比：
‌Precision‌: 预测为正样本中实际为正的比例（计算公式：TP / (TP + FP)）
‌Recall‌: 实际正样本中被正确预测的比例（计算公式：TP / (TP + FN)）
‌F1-score‌: Precision和Recall的调和平均值（计算公式：2 × (Precision × Recall) / (Precision + Recall)），用于平衡两者的性能
Support：该类别的真实样本数量
accuracy: 为整体准确率
‌宏平均 (macro avg)‌：所有类别的指标算术平均值（不考虑样本数量）
‌加权平均 (weighted avg)‌：按各类别样本数量加权计算的指标平均值
'''