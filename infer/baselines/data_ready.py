#  @file: data_ready.py
#  @version：1.0.0
#  @brief: Ready data for baseline testing
#  @creation date: 2026.10.14
#  @last modified date: 2026.10.15 
#  @authors: S. Yang
#  @copyright: ©2026 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


import numpy as np

original_file = r'original.csv'
data = []
with open(original_file, 'r', encoding='utf-8') as f:
    next(f) # skip header
    data = f.readlines()
    # print(len(data))


ready_data = []   # like (id, project, description, type)
type_transfer = {
    '政府主导': 'A',
    '社区主导': 'B',
    '其他类型': 'C',
}
for i in range(0, len(data)):
    line = data[i].strip().split(',')
    ready_data.append((i+1, line[1], line[2], type_transfer[line[7]]))

# describe dataset    
print("All Samples:", len(ready_data))
A = 0
B = 0
C = 0
description_length = []
A_length = []
B_length = []
C_length = []
for i in range(len(ready_data)):
    description_length.append(len(ready_data[i][2]))
    if ready_data[i][3] == 'A':
        A += 1
        A_length.append(len(ready_data[i][2]))
    elif ready_data[i][3] == 'B':
        B += 1
        B_length.append(len(ready_data[i][2]))
    elif ready_data[i][3] == 'C':
        C += 1
        C_length.append(len(ready_data[i][2]))
print("  type A:", A)
print("  type B:", B)
print("  type C:", C)
print("Avg. description length:", int(np.mean(description_length)))
print("  Avg. descr. of type A:", int(np.mean(A_length)))
print("  Avg. descr. of type B:", int(np.mean(B_length)))
print("  Avg. descr. of type C:", int(np.mean(C_length)))

# save to file data.csv
with open('data.csv', 'w', encoding='utf-8') as f:
    for item in ready_data:
        f.write(','.join([str(i) for i in item]) + '\n')
        
