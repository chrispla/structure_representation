#Importing
import librosa
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp2d
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import directed_hausdorff
from scipy.cluster import hierarchy
from scipy.linalg import eigh
from scipy.ndimage import median_filter
import cv2
from sklearn import metrics
import matplotlib.pyplot as plt
import dill
import sys
import glob
import os
import random
from segment_transformation import segment


#--supress warnings--#
import warnings
warnings.filterwarnings("ignore")


#--reading--#

all_dirs = []
all_names = []
all_roots = []
max_files = 4000

#load directories of original, non-transformed songs
for root, dirs, files in os.walk('/home/ismir/Documents/ISMIR/Datasets/covers80-perturbations/'):
        for name in files:
            if ('.mp3' in name):
                filepath = os.path.join(root, name)
                all_dirs.append(filepath)
                all_names.append(name[:-4])
                all_roots.append(root)
                if len(all_dirs)>=max_files:
                    break
        if len(all_dirs)>=max_files:
            break        
file_no = len(all_dirs)

#--Structure dictionary--#
#format: struct[song name][transformation] = [struct, flat, merged, shingled2, shingled3]
struct = {}
for name in all_names:
    struct[name] = {}
    struct[name]['OG'] = [] #entry for original song's structure
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        struct[name][tf]=[]


#--Distance dictionary--#
#format: dist[metric][name][transform] = float
#metrics: L1, fro, hau, dtw, pair, sh2, sh3
dist = {}
for metric in ['L1', 'fro', 'hau', 'dtw', 'pair', 'sh2', 'sh3']:
    dist[metric] = {}
    for name in all_names:
        dist[metric][name] = {}
        for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
            dist[metric][name][tf]=0


#segment songs
kmin = 2
kmax = 11
rs_size = 128 #resampling size for combined matrix
tf_no = 17 #number of transformations per file

#for original audio
count = 0

for f in range(file_no):

    #segment original

    #structure segmentation
    approximations = segment(all_dirs[f], rs_size, kmin, kmax, True)
    struct[all_names[f]]['OG'].append(approximations)

    #formatting
    flat_approximations = []
    merged_approximations = np.empty((0))
    for j in range(kmax-kmin):
        flat_approximations.append(approximations[j].flatten())
        merged_approximations = np.concatenate((merged_approximations, flat_approximations[j]))
    struct[all_names[f]]['OG'].append(np.asarray(flat_approximations))
    struct[all_names[f]]['OG'].append(merged_approximations)

    #shingling per 2
    shingled = []
    for j in range(kmax-kmin-1):
        #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1]]))
        shingled.append(np.concatenate((struct[all_names[f]]['OG'][1][j],
                                        struct[all_names[f]]['OG'][1][j+1]), 
                                        axis=None))
    struct[all_names[f]]['OG'].append(np.asarray(shingled))

    #shingling per 3
    shingled = []
    for j in range(kmax-kmin-2):
        #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1],all_flat[f][j+2]]))
        shingled.append(np.concatenate((struct[all_names[f]]['OG'][1][j],
                                        struct[all_names[f]]['OG'][1][j+1],
                                        struct[all_names[f]]['OG'][1][j+2]), 
                                        axis=None))
    struct[all_names[f]]['OG'].append(np.asarray(shingled))
    
    count+=1

    #segments transformations
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:

        #construct filedir of transformations from filedir of original, and segment
        approximations = segment(all_roots[f] + '/'+ tf + '-' + all_names[f] + '.wav', rs_size, kmin, kmax, True)
        struct[all_names[f]][tf].append(approximations)

        #formatting
        flat_approximations = []
        merged_approximations = np.empty((0))
        for j in range(kmax-kmin):
            flat_approximations.append(approximations[j].flatten())
            merged_approximations = np.concatenate((merged_approximations, flat_approximations[j]))
        struct[all_names[f]][tf].append(np.asarray(flat_approximations))
        struct[all_names[f]][tf].append(merged_approximations)

        #shingling per 2
        shingled = []
        for j in range(kmax-kmin-1):
            #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1]]))
            shingled.append(np.concatenate((struct[all_names[f]][tf][1][j],
                                            struct[all_names[f]][tf][1][j+1]), 
                                            axis=None))
        struct[all_names[f]][tf].append(np.asarray(shingled))

        #shingling per 3
        shingled = []
        for j in range(kmax-kmin-2):
            #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1],all_flat[f][j+2]]))
            shingled.append(np.concatenate((struct[all_names[f]][tf][1][j],
                                            struct[all_names[f]][tf][1][j+1],
                                            struct[all_names[f]][tf][1][j+2]), 
                                            axis=None))
        struct[all_names[f]][tf].append(np.asarray(shingled))

    #progress
    sys.stdout.write("\rSegmented %i/%s pieces." % ((f*tf_no)+count, str(file_no*(tf_no+1))))
    sys.stdout.flush()

print('')


#figure directory
fig_dir = '/home/ismir/Documents/ISMIR/figures/deformations_run2/'

#L1 norm
for name in all_names:
    #traverse transformations
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        dist['L1'][name][tf] = np.linalg.norm(struct[name]['OG'][2]-struct[name][tf][2], ord=1) #2->merged
print("Computed L1 distances.")

#Frobenius norm
for name in all_names:
    #traverse transformations
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        dist['fro'][name][tf] = np.linalg.norm(struct[name]['OG'][2]-struct[name][tf][2])
print("Computed Frobenius distances.")

#Sub-sequence Dynamic Time Warping cost
for name in all_names:
    #traverse transformations
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        costs = []
        for k in range(kmax-kmin):
            costs.append(librosa.sequence.dtw(struct[name]['OG'][0][k], #0->original structure format
                                                struct[name][tf][0][k], 
                                                subseq=True, 
                                                metric='euclidean')[0][rs_size-1,rs_size-1])
        dist['dtw'][name][tf] = sum(costs)/len(costs)
print("Computed DTW cost.")

#Directed Hausdorff distance
for name in all_names:
    #traverse transformations
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        dist['hau'][name][tf] = (directed_hausdorff(struct[name]['OG'][1], struct[name][tf][1]))[0] #1->flat
print("Computed directed Hausdorff distances.")

#Minimum distance across all pairs
for name in all_names:
    #traverse transformations
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        dists = []
        for n in range(kmax-kmin):
            for m in range(kmax-kmin):
                dists.append(np.linalg.norm(struct[name]['OG'][0][n]-struct[name][tf][0][m])) #0->original structure format
        dist['pair'][name][tf] = min(dists)
print("Computed minimum pairwise distance.")

#Directed Hausdorff distance shingled tuples
for name in all_names:
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        dist['sh2'][name][tf] = (directed_hausdorff(struct[name]['OG'][3], struct[name][tf][3]))[0] #3->shingled2
print("Computed directed Hausdorff distances for bi-grams.")

#Directed Hausdorff distance shingled triples
for name in all_names:
    for tf in ['T03S', 'T07S', 'T15S', 'T03E', 'T07E', 'T15E',
                'S03S', 'S07S', 'S15S', 'S03E', 'S07E', 'S15E',
                'SWAP', 'REM1', 'REM2', 'DUP1', 'DUP2']:
        dist['sh2'][name][tf] = (directed_hausdorff(struct[name]['OG'][4], struct[name][tf][4]))[0] #3->shingled3
print("Computed directed Hausdorff distances for tri-grams.")


dill.dump_session('/home/ismir/Documents/ISMIR/dills/deformations_run2/deformations.db')