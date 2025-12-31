#  @file: ahplot.py
#  @version：1.0.5
#  @brief: Compute Eigenvalues and Eigenvectors of a matrix in Excel
#  @creation date: 2025.10.11
#  @last modified date: 2025.11.11 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 
 


import numpy as np
import xlwings as xw
import os
from typing import Annotated

Array2d = Annotated[np.ndarray, {"ndim": 2}]
List2d = Annotated[list[list[float]], {"ndim": 2}]

import pandas as pd
Df = Annotated[pd.DataFrame, {"index": False}]
@xw.func  # 定义一个Excel函数，在表格编辑中使用
# @xw.arg("df", index=False)
# @xw.ret(index=False)
def myfunction(df: Df) -> Df:
    # df is a DataFrame, do something with it
    return df

# 从外部打开Excel文件
wb = xw.Book(r'weights.xlsm')  
sheet = wb.sheets[0]  # 获取Excel文件中的第一个工作表

@xw.sub  # 定义一个子过程在VBA中调用
def eigen_sub(data_range:str, eigenvalues_range:str, eigenvectors_range:str):
    wb = xw.Book.caller()  # 获取调用函数的Excel文件
    sheet = wb.sheets[0]  # 获取Excel文件中的第一个工作表
    # 获取Excel文件中的数据
    data = sheet.range(data_range).value  
    eigenvalues, eigenvectors = np.linalg.eig(data)

    # 将结果写入Excel文件
    sheet.range(eigenvalues_range).options(transpose=True).value = eigenvalues.real  # 将特征值写入Excel文件
    sheet.range(eigenvectors_range).options(transpose=True).value = eigenvectors.real  # 特征向量是列向量，第一列是第一个特征值的特征向量

    # wb.save()  # 保存Excel文件
    # wb.close()  # 关闭Excel文件

@xw.func
def add_one(data:List2d):  # data is a range
    return [[cell + 10 for cell in row] for row in data]  # excel编辑状态函数，不能是vba调用的函数
    # return [[sheet.range(cell).value + 1  for cell in row] for row in data]

@xw.sub  # 这也是一个子过程，在VBA中调用
def add_one_sub(data_range:str, result_range:str):
    wb = xw.Book.caller()  # 获取调用函数的Excel文件
    sheet = wb.sheets[0]  # 获取Excel文件中的第一个工作表
    data = sheet.range(data_range).value
    result = [[cell + 1 for cell in row] for row in data]
    sheet.range(result_range).value = result

@xw.func
def directory():
    return os.getcwd()

if __name__ == '__main__':
    # print('We are working here: ', directory())
    # matrix = Split("C12:F15,C21:F24,C30:F33,C39:F42", ",")
    # eigenvalues = Split("S12:S15,S21:S24,S30:S33,S39:S42", ",")
    # eigenvectors = Split("T12:W15,T21:W24,T30:W33,T39:W42", ",")
    # eigen_sub('C12:F15', 'S12:S15', 'T12:W15')  # 作为外部函数调用不能用 wb = xw.Book.caller()
    pass
