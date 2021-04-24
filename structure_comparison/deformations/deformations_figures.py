#Importing
import librosa
import numpy as np
import matplotlib.pyplot as plt
import dill
import sys
import glob
import os
import csv

dill.load_session('../../../dills/deformations_all.db')

tfs = [] #list of all possible transformations
for edit in ['T', 'S']: #for edit in Trim, Silence
    for position in ['S', 'E']: #for position in Start, End
        for duration in ['03', '07', '15']: #for duration in 3sec, 7sec, 15sec
                tfs.append(edit+duration+position)

rows = [['']] #0,0 position empty
for tf in tfs: 
    rows[0].append(tf)
for metric in ['L1', 'fro', 'dtw', 'hau', 'pair']:
    row = [metric]
    for tf in tfs:
        distance = 0
        for name in all_names:
            distance+=dist[metric][name][tf]
        distance = distance/float(file_no)
        row.append(distance)
    rows.append(row)

with open('/Users/chris/Google Drive/Classes/Capstone/figures/deformations/distances.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

