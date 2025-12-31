#  @file: dimensionalSimilarity.py
#  @version：1.0.5
#  @brief: # Computes the psychological similarity between two individuals based on their psychological dimensions.
#  @creation date: 2025.08.28
#  @last modified date: 2025.10.12 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 


from projcondSimilarity import *

one = 'bai'
two = 'hani'

simi = psychology_similarity(one, two)

print(simi)