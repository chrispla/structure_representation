#Importing
import librosa
import numpy as np
import matplotlib.pyplot as plt
import dill
import sys
import glob
import os
import csv

dill.load_session('/home/ismir/Documents/ISMIR/dills/deformations_run2/deformations.db')

#list of all possible transformations
tfs =  ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']

rows = [['']] #0,0 position empty
for tf in tfs: 
    rows[0].append(tf)
for metric in ['L1', 'fro', 'dtw', 'hau', 'pair', 'sh2', 'sh3']:
    row = [metric]
    for tf in tfs:
        distance = 0
        for name in all_names:
            distance+=dist[metric][name][tf]
        distance = distance/float(file_no)
        row.append(distance)
    rows.append(row)

with open('/home/ismir/Documents/ISMIR/figures/deformations_run2/perturbations.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

