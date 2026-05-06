#  @file: decisiontree.py
#  @version：1.0.0
#  @brief:  Trying to use decision tree to classify the data for comparison
#  @creation date: 2026.04.23
#  @last modified date: 2026.04.23 
#  @authors: S. Yang
#  @copyright: © 2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns  

from pathlib import Path
pwd = Path(__file__).resolve().parent

file = pwd / 'assesmentvalue++.csv'  # modify the input filename to load different datasets
assessvalue = np.loadtxt(file, dtype=float, delimiter=',')   
print(assessvalue.shape)

X = assessvalue[:, :-1]   # features
y = assessvalue[:, -1]    # labels
labels = ['community-led', 'government-led', 'mixed']  # (0,1,2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier(criterion='gini', random_state=42)
dtc.fit(X_train, y_train)


# Print the evaluation report on test data

y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=labels))
print(f"Accuracy of Decision Tree Classifier: {accuracy:.4f}")
cm = confusion_matrix(y_test, y_pred)


# Print the evaluation report on all data

# y_pred = dtc.predict(X)
# accuracy = accuracy_score(y, y_pred)
# print(classification_report(y, y_pred, target_names=labels))
# print(f"Accuracy of Decision Tree Classifier: {accuracy:.4f}")
# cm = confusion_matrix(y, y_pred)


# Visualize the confusion matrix by the decision tree classifier

plt.figure(figsize=(6, 4.2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



'''
(38, 6)  只包含测试数据，有意义
                precision    recall  f1-score   support

 community-led       0.92      0.92      0.92        24
government-led       0.50      0.75      0.60         4
         mixed       0.88      0.70      0.78        10

      accuracy                           0.84        38
     macro avg       0.76      0.79      0.76        38
  weighted avg       0.86      0.84      0.85        38

Accuracy of Decision Tree Classifier: 0.8421
'''

'''
(188, 6)  包含了训练数据，意义不大
                precision    recall  f1-score   support

 community-led       0.98      0.98      0.98       113
government-led       0.91      0.97      0.94        30
         mixed       0.98      0.93      0.95        45

      accuracy                           0.97       188
     macro avg       0.96      0.96      0.96       188
  weighted avg       0.97      0.97      0.97       188

Accuracy of Decision Tree Classifier: 0.9681
'''