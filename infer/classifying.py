# KNN Algorithm with k = 3 (goverment-led, community-leb, mixed pattern)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

class KNNClassifier:
    def __init__(self, k=3):
        """
        参数:
        k (int): 最近邻的数量，默认为3
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        """
        计算两个样本点之间的欧氏距离
        
        参数:
        x1 (numpy.ndarray): 第一个样本点
        x2 (numpy.ndarray): 第二个样本点
        
        返回:
        float: 两个样本点之间的欧氏距离
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X, y):
        """
        训练模型，保存训练数据
        
        参数:
        X (numpy.ndarray): 训练特征数据
        y (numpy.ndarray): 训练标签数据
        """
        self.X_train = X
        self.y_train = y
    
    def predict_single(self, x):
        """
        预测单个样本的类别
        
        参数:
        x (numpy.ndarray): 待预测样本
        
        返回:
        int: 预测的类别标签
        """
        # 计算与训练集中所有样本的距离
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 获取最近的K个邻居的索引
        k_indices = np.argsort(distances)[:self.k]
        
        # 获取最近的K个邻居的类别标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 多数表决确定预测类别
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """
        预测多个样本的类别
        
        参数:
        X (numpy.ndarray): 待预测样本集
        
        返回:
        list: 预测的类别标签列表
        """
        return np.array([self.predict_single(x) for x in X])

def plot_decision_boundary(model, X, y, title="KNN classification boundary"):
    """
    绘制KNN分类器的决策边界和分类结果
    
    参数:
    model: 训练好的KNN模型
    X (numpy.ndarray): 特征数据
    y (numpy.ndarray): 标签数据
    title (str): 图表标题
    """
    # 创建网格点
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 创建颜色映射
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    plt.figure(figsize=(12, 8))
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # 绘制训练数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                edgecolor='black', s=50, alpha=0.8)
    
    plt.xlabel('Scores', fontsize=12)
    plt.ylabel('Aesthetic Ontology', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 添加图例
    classes = ['Class 0', 'Class 1', 'Class 2']
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=cmap_bold(i), markersize=8) 
               for i in range(3)]
    plt.legend(handles, classes, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def main():
    # 1. 读取数据集
    print("1. 读取数据集...")
    assessvalue = np.loadtxt(r'assesmentvalue++.csv', dtype=float, delimiter=',')  # TODO: 修改数据集路径
    print(assessvalue.shape)  # 注意数据的列数

    # 这里是真实数据集，可以直接使用
    target_cols =[0, 3]  # TODO: 修改特征列
    X, y = assessvalue[:, target_cols], assessvalue[:, -1]  # 仅使用前两列特征，最后一列为标签
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))   # 数据标准化
    print(X.shape, y.shape)
    
    # 2. 数据集分割
    print("2. 数据集分割...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. 数据标准化
    print("3. 数据Z标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 创建并训练KNN分类器
    print("4. 训练KNN分类器...")
    K = 7  # TODO: 调整K值, 通常采用K-Cross Validation选择最优K值，也可以直接用sklearn的GridSearchCV方法
    knn = KNNClassifier(k=K)  # k值不是类别数，而是邻居数，需要自己调整（超参数选择）
    knn.fit(X_train_scaled, y_train)
    
    # 5. 进行预测
    print("5. 进行预测...")
    y_pred = knn.predict(X_test_scaled)
    
    # 6. 模型评估
    print("6. 模型评估结果:")
    print("-" * 30)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 7. 绘制分类结果图
    print("7. 绘制分类结果图...")
    
    # 创建子图布局
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6.4, 4.8))
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(5.2, 3.8))
    
    # 图1：原始数据分布
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                     cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), 
                     edgecolor='black', s=50)
    ax1.set_title('The original distribution of the scores', fontsize=12, fontweight='bold')  # 原始训练数据分布
    ax1.set_xlabel('Scores')
    ax1.set_ylabel('Aesthetic Ontology')
    ax1.grid(True, alpha=0.3)
    ax1.legend(*scatter1.legend_elements(), 
               title='Classes', loc='lower right')
    
    # 图2：KNN决策边界
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax2.contourf(xx, yy, Z, 
                 cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), 
                 alpha=0.8)
    scatter2 = ax2.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                     c=y_train, 
                     cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), 
                     edgecolor='black', s=50)
    ax2.set_title(f'KNN Decision Boundary (K={K})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Scores (Normalized)')
    ax2.set_ylabel('Aesthetic Ontology (Normalized)')
    ax2.grid(True, alpha=0.3)
    
    # # 图3：测试集预测结果
    # scatter3 = ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
    #                  cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), 
    #                  edgecolor='black', s=50, alpha=0.7, label='real')
    # scatter4 = ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, 
    #                  cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), 
                    #  marker='x', s=80, label='predict')
    # ax3.set_title('Comparison between predicted and true values', fontsize=12, fontweight='bold')  # 预测结果对比
    # ax3.set_xlabel('Scores')
    # ax3.set_ylabel('Aesthetic Ontology')
    # ax3.legend(loc='lower right')
    # ax3.grid(True, alpha=0.3)
    
    # 图4：混淆矩阵热力图
    # cm = confusion_matrix(y_test, y_pred)
    # im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # # 添加混淆矩阵数值标注
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax4.text(j, i, format(cm[i, j], 'd'),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    
    # ax4.set_xlabel('Predicted Label')
    # ax4.set_ylabel('True Label')
    # ax4.set_xticks(range(3))
    # ax4.set_yticks(range(3))
    # ax4.set_xticklabels(['0', '1', '2'])
    # ax4.set_yticklabels(['0', '1', '2'])
    
    plt.tight_layout()
    plt.show()
    
    # 8. 分析KNN算法的优缺点
    print("\nKNN算法特点分析:")
    print("- 优点: 思想简单，易于理解和实现，对异常值不敏感")
    print("- 缺点: 计算量大，需要存储整个训练数据集")
    print("- 适用场景: 小数据场景，几千~几万样本")

if __name__ == "__main__":
    main()


