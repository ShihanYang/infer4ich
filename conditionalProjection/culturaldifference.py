#  @file: culturaldifference.py
#  @version：1.0.5
#  @brief: # This file is used to calculate the cultural difference between two groups of people 
#            by using the similarity of their intangible cultural heritage features.
#  @last modified date: 2025.10.12 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 



from projcondSimilarity import *

one = 'bai'
two = 'hani'

simi = culture_difference(one, two)

print(simi)