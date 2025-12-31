#  @file: metrics.py
#  @version：1.0.5
#  @brief: Evaluate the precision, recall, and F1-score of predicted values relative to true values.
#  @creation date: 2025.08.28
#  @last modified date: 2025.11.12 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


from sklearn.metrics import classification_report
from infer.classifying import KNNClassifier as knn
import numpy as np

test_data = np.loadtxt(r'assesmentvalue++.csv', dtype=float, delimiter=',')
X_test = test_data[:, :-1]

y_true = test_data[:,-1]  # 真实标签
y_pred = knn().predict(X_test)  # 预测标签

report = classification_report(y_true, y_pred)

print(report)

