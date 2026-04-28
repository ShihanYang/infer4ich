#  @file: culturaldifference.py
#  @version：1.1.0
#  @brief: # This file is used to calculate the cultural difference (distance) between two groups of people 
#            by using the similarity (non-conditional) of their intangible cultural heritage features.
#          # Note: maybe we need to choose the distance metrics.
#  @creation date: 2025.08.28 
#  @last modified date: 2026.04.22 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 

 

from projcondSimilarity import *

one = 'bai'  # bai-syj 
two = 'hani'  # hani-jzsl
three = 'lisu'  # lisu-dgj

simi12 = culture_difference(one+'-syj', two+'-jzsl')
print(simi12)

simi13 = culture_difference(one+'-syj', three+'-dgj')
print(simi13)

simi23 = culture_difference(two+'-jzsl', three+'-dgj')
print(simi23)
