#  @file: culturaldifference.py
#  @version：1.1.0
#  @brief: # This file is used to calculate the cultural difference between two groups of people 
#            by using the similarity of their intangible cultural heritage features.
#  @last modified date: 2026.04.22 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 



from conditionalProjection.projcondSimilarity import *

one = 'bai'
two = 'hani'
three = 'lisu'

simi12 = culture_difference(one, two)
print(simi12)

simi13 = culture_difference(one, three)
print(simi13)

simi23 = culture_difference(two, three)
print(simi23)
