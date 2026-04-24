#  @file: weighted_scores.py
#  @version：1.0.8
#  @brief: computing the weighted scores in following weight-vectors
#          disciplinary weights on the cultural psychological framework
#  @creation date: 2025.08.28
#  @last modified date: 2026.04.24
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 



####  pyschological dimension      #    disciplinary weights (history/Aesthetic/semiology/sociology) ###
# ------------------------------------------------------------------------------------------------------------
#    independent/interdependent    #       (0.5150,0.1379,0.0392,0.3079)
#    individualism/collectivism    #       (0.4393,0.0508,0.1247,0.3852)
#      tightness/looseness         #       (0.3425,0.0901,0.0574,0.5101)
#      relational mobility         #       (0.5672,0.0461,0.1323,0.2544)
# ------------------------------------------------------------------------------------------------------------

import numpy as np
from projection2 import loadEmbedding, project_vector_set, score, score2
from pathlib import Path
pwd = Path(__file__).resolve().parent


# the embedding of word lists in each pyschological dimension
ws_embeddings = dict()   # dict_keys = (independ, interdepend, individ, collect, tight, loose, relmobility)

# vector spaces of the four disciplines 
spaces = dict()   # dict_keys = (history, aesthetic, semiology, sociology)

# disciplinary weights for each pyschological dimension from AHP in excel
on_independ_interdepend = (0.5150,0.1379,0.0392,0.3079)
on_individ_collect = (0.4393,0.0508,0.1247,0.3852)
on_tight_loose = (0.3425,0.0901,0.0574,0.5101)
on_relmobility = (0.5672,0.0461,0.1323,0.2544)

# Specified assessment object
assessed_name = 'lisu'  # TODO: Changing the assessed name, 'bai',  'hani' or 'lisu'

# loading embedding
space_files = [
    "History.csv.vec",
    "Aesthetic.csv.vec",
    "Semiology.csv.vec",
    "Sociology.csv.vec"    
]
wordset_files = [
    "independ.csv.vec",
    "interdepend.csv.vec",
    "individ.csv.vec",
    "collect.csv.vec",
    "tight.csv.vec",
    "loose.csv.vec",
    "relmobility.csv.vec"
]

# load embedding domain spaces
directory = pwd / "data"
for sf in space_files:
    key = sf.split('.')[0].lower()
    spaces[key] = loadEmbedding(directory / sf)

print('Disciplinary Space:')
print('      history -', spaces['history'].shape)
print('    aesthetic -', spaces['aesthetic'].shape)
print('    semiology -', spaces['semiology'].shape)
print('    sociology -', spaces['sociology'].shape)

# load word sets embedding
directory = directory / assessed_name
for wsf in wordset_files:
    key = wsf.split('.')[0].lower()
    ws_embeddings[key] = loadEmbedding(directory / wsf)

print('Wordsets Embedding:')    
for k in ws_embeddings.keys():
    print('    ', k, '-', ws_embeddings[k].shape)
    
spacename = ['history', 'aesthetic', 'semiology', 'sociology']
dimension = ['independ', 'interdepend', 'individ', 'collect', 'tight', 'loose', 'relmobility']


################################################
# 投影 projecting each dimension onto each space
################################################
projections = dict()
for space in spacename:
    for dim in dimension:
        projections[(dim, space)] = project_vector_set(ws_embeddings[dim], spaces[space])


################################################
# 计算各认知维度在各空间上投影的加权得分
################################################
from datetime import datetime
print(f"{assessed_name.upper()} @ {datetime.now()}")
print(f"Got scores of {assessed_name.upper()} on weighted disciplines :")

scores = dict()
for key, prj in projections.items():
    scores[key] = score(prj)   # 注意选取不同的score函数，得到不同的结果, TODO: Changing the score methods
    # scores[key] = score2(prj)  # TODO: Changing the score methods

########################
# dimension 1
########################
print('- independ:')
independ = (scores[('independ', 'history')], 
            scores[('independ', 'aesthetic')],
            scores[('independ', 'semiology')],
            scores[('independ', 'sociology')])
weighted_score = np.dot(on_independ_interdepend, independ)
print('  ', weighted_score)  # weighted score

print('- interdepend:')
interdepend = (scores[('interdepend', 'history')],  
               scores[('interdepend', 'aesthetic')],
               scores[('interdepend', 'semiology')],
               scores[('interdepend', 'sociology')])
weighted_score = np.dot(on_independ_interdepend, interdepend)
print('  ', weighted_score) 

########################
# dimension 2
########################
print('- individ:')
individ = (scores[('individ', 'history')], 
           scores[('individ', 'aesthetic')],
           scores[('individ', 'semiology')],
           scores[('individ', 'sociology')])
weighted_score = np.dot(on_individ_collect, individ)
print('  ', weighted_score)

print('- collect:')
collect = (scores[('collect', 'history')], 
           scores[('collect', 'aesthetic')],
           scores[('collect', 'semiology')],
           scores[('collect', 'sociology')])
weighted_score = np.dot(on_individ_collect, collect)
print('  ', weighted_score)

########################
# dimension 3
########################
print('- tight:')
tight = (scores[('tight', 'history')], 
         scores[('tight', 'aesthetic')],
         scores[('tight', 'semiology')],
         scores[('tight', 'sociology')])
weighted_score = np.dot(on_tight_loose, tight)
print('  ', weighted_score)

print('- loose:')
loose = (scores[('loose', 'history')], 
         scores[('loose', 'aesthetic')],
         scores[('loose', 'semiology')],
         scores[('loose', 'sociology')])
weighted_score = np.dot(on_tight_loose, loose)
print('  ', weighted_score)

########################
# dimension 4
########################
print('- relmobility:')
relmobility = (scores[('relmobility', 'history')], 
               scores[('relmobility', 'aesthetic')],
               scores[('relmobility', 'semiology')],
               scores[('relmobility', 'sociology')])
weighted_score = np.dot(on_relmobility, relmobility)
print('  ', weighted_score)

