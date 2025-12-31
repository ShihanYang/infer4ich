#  @file: weights.py
#  @version：1.0.5
#  @brief: Compute Eigenvalues and Eigenvectors of a matrix in Excel
#  @creation date: 2025.10.11
#  @last modified date: 2025.11.11 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 



import numpy as np
import xlwings as xw

# 打开Excel文件
wb = xw.Book.caller()  # 在excel中运行时使用

sheet = wb.sheets[0]  # 获取Excel文件中的第一个工作表
data = sheet.range("C12:F15").value  # 获取Excel文件中的数据"C12:F15")  
print(data)
eigenvalues, eigenvectors = np.linalg.eig(data)

# 输出结果
print("Eigenvalues:", eigenvalues)  # 特征值是复数，这里只取实部
print("Eigenvectors:", eigenvectors)  # 特征向量

# 将结果写入Excel文件
sheet.range("S12:S15").options(transpose=True).value = eigenvalues.real  # 将特征值写入Excel文件
sheet.range("T12:W15").options(transpose=True).value = eigenvectors.real  # 特征向量是列向量，第一列是第一个特征值的特征向量

# wb.save()  # 保存Excel文件
# wb.close()  # 关闭Excel文件


def main():
    return  0

if __name__ == "__main__":
    main()

