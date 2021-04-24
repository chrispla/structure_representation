#Importing
import librosa
import numpy as np
import matplotlib.pyplot as plt
import dill
import sys
import glob
import os
import random
import csv

dill.load_session('../../../dills/covers80_all.db')

"""Terminology
distances = {}
distances: L1, fro, dtw, hau, pair
format: (filt-)rs_size-approx[0]-approx[1]-distance e.g. filt-128-2-8-L1
"""

with open('/Users/chris/Google Drive/Classes/Capstone/figures/covers80_run1/distances.csv', mode='w') as f:
    writer = csv.writer(f)
    for metric in ['L1', 'fro', 'dtw', 'hau', 'pair']:
        for filtering in ['filt-', '']:
            row = [filtering+metric]
            for approx in ['[3-7]', '[4-10]', '[8-11]']:
                row.append(approx)
            #write first row
            writer.writerow(row)
            for resamp in ['64', '128']:
                row = [resamp]
                for approx in ['3-7', '4-10', '8-11']:
                    key = filtering+resamp+'-'+approx+'-'+metric
                    row.append(scores[key])
                writer.writerow(row)
        #space before new table
        writer.writerow([''])




# #Plotting scores
# fig, ax = plt.subplots()
# xticks = list(scores.keys())
# all_scores = list(scores.values())

# ax.barh(xticks, all_scores)
# ax.set(xlabel='Score', ylabel='Construction')
# plt.savefig('bar_plot.png')
