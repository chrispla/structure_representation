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

dill.load_session('/home/chris/Documents/dills/5covers80_all.db')

"""Terminology
distances = {}
distances: L1, fro, dtw, hau, pair, sh2, sh3
format: (filt-)rs_size-approx[0]-approx[1]-distance e.g. filt-128-2-8-L1
"""

with open('/home/chris/Documents/figures/distances.csv', mode='w') as f:
    writer = csv.writer(f)
    for metric in ['L1', 'fro', 'dtw', 'hau', 'pair', 'sh2', 'sh3']:
        for filtering in ['filt-', '']:
            row = [filtering+metric]
            for approx in ['[2-5]', '[2-7]', '[4-9]', '[8-11]']:
                row.append(approx)
            #write first row
            writer.writerow(row)
            for resamp in ['32', '64', '128', '256']:
                row = [resamp]
                for approx in ['2-5', '2-7', '4-9', '8-11']:
                    key = filtering+resamp+'-'+approx+'-'+metric
                    row.append(scores[key])
                writer.writerow(row)
        #space before new table
        writer.writerow([''])