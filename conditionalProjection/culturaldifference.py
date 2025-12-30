# This file is used to calculate the cultural difference between two groups of people 
# by using the similarity of their intangible cultural heritage features.

from projcondSimilarity import *

one = 'bai'
two = 'hani'

simi = culture_difference(one, two)

print(simi)