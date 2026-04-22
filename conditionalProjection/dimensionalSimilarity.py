#  @file: dimensionalSimilarity.py
#  @version：1.1.0
#  @brief: # Computes the psychological similarity between two individuals based on their psychological dimensions.
#  @creation date: 2025.08.28
#  @last modified date: 2024.04.22 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


from conditionalProjection.projcondSimilarity import *

one = 'bai'
two = 'hani'
three = 'lisu'

simi12 = psychology_similarity(one, two)
print(simi12)

simi13 = psychology_similarity(one, three)
print(simi13)

simi23 = psychology_similarity(two, three)
print(simi23)
