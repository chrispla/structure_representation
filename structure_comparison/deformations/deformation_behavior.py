def segment(filedir, rs_size, kmin, kmax, filter):
    """structurally segments the selected audio

        ds_size: side length to which combined matrix is going to be resampled to
        [kmin, kmax]: min and maximum approximation ranks

    returns set of low rank approximations"""

    #load audio
    y, sr = librosa.load(filedir, sr=16000, mono=True)

        #compute cqt
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, 
                                        hop_length=512,
                                        bins_per_octave=12*3,
                                        n_bins=7*12*3)),
                                        ref=np.max)

    #beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)

    #beat synch cqt
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    #stack memory
    if filter:
        Csync = librosa.feature.stack_memory(Csync, 4)

    #Affinity matrix
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)

    #Filtering
    if filter:  
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        R = df(R, size=(1, 7))
        R = librosa.segment.path_enhance(R, 15)

    #mfccs
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    #beat sync mfccs
    Msync = librosa.util.sync(mfcc, beats)

    #weighted sequence
    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    #weighted combination of affinity matrix and mfcc diagonal
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(R, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * R + (1 - mu) * R_path

    #resampling
    A_d = cv2.resize(A, (rs_size, rs_size))

    #laplacian
    L = scipy.sparse.csgraph.laplacian(A_d, normed=True)

    #eigendecomposition
    evals, evecs = scipy.linalg.eigh(L)
    #eigenvector filtering
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    #normalization
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    #temporary replacement for bug
    a_min_value = 3.6934424e-08
    Cnorm[Cnorm == 0.0] = a_min_value
    if (np.isnan(np.sum(Cnorm))):
        print("WOOOOOAH")

    #approximations
    dist_set = []
    for k in range(kmin, kmax):

        Xs = evecs[:, :k] / Cnorm[:, k-1:k]
        
        #debug
        if np.isnan(np.sum(Xs)):
            print('woops')

        distance = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Xs, metric='euclidean'))
        dist_set.append(distance)
    dist_set = np.asarray(dist_set)
    
    
    #return
    return(dist_set)

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

#--supress warnings--#
import warnings
warnings.filterwarnings("ignore")


#--reading--#

all_dirs = []
all_names = []
all_roots = []
max_files = 4000

#load directories of original, non-transformed songs
for root, dirs, files in os.walk('/Users/chris/Google Drive/Classes/Capstone/Datasets/deformations/'):
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
#format: struct[song name][transformation] = [struct, flat, merged]
struct = {}
for name in all_names:
    struct[name] = {}
    struct[name]['OG'] = [] #entry for original song's structure
    for edit in ['T', 'S']: #for edit in Trim, Silence
        for duration in ['03', '07', '15']: #for duration in 3sec, 7sec, 15sec
            for position in ['S', 'E']: #for position in Start, End
                struct[name][edit+duration+position]=[]


#--Distance dictionary--#
#format: dist[metric][name][transform] = float
#metrics: L1, fro, hau, dtw, pair
dist = {}
for metric in ['L1', 'fro', 'hau', 'dtw', 'pair']:
    dist[metric] = {}
    for name in all_names:
        dist[metric][name] = {}
        for edit in ['T', 'S']: #for edit in Trim, Silence
            for duration in ['03', '07', '15']: #for duration in 3sec, 7sec, 15sec
                for position in ['S', 'E']: #for position in Start, End
                    dist[metric][name][edit+duration+position]=0


#segment songs
kmin = 7
kmax = 11
rs_size = 64 #resampling size for combined matrix
tf_no = 12 #number of transformations per file

#for original audio
for f in range(file_no):

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
    
    count = 0
    #traverse transformations
    for edit in ['T', 'S']: #for edit in Trim, Silence
        for duration in ['03', '07', '15']: #for duration in 3sec, 7sec, 15sec
            for position in ['S', 'E']: #for position in Start, End

                tf = edit+duration+position #transformation string

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

                count+=1

    #progress
    sys.stdout.write("\rSegmented %i/%s pieces and their transformations." % ((f*tf_no)+count+1, str(file_no*tf_no)))
    sys.stdout.flush()

print('')


#figure directory
fig_dir = '/Users/chris/Google Drive/Classes/Capstone/figures/deformations/'





exit()





#L1 norm
L1_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        L1_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j], ord=1)

key = 'L1'
distances[key] = L1_distances


print("Computed L1 distances.")

#Frobenius norm
fro_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        fro_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j])
key = 'fro'
distances[key] = fro_distances
       

print("Computed Frobenius distances.")

#Sub-sequence Dynamic Time Warping cost
dtw_cost = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        costs = []
        for k in range(kmax-kmin):           
            costs.append(librosa.sequence.dtw(all_struct[i][k], all_struct[j][k], subseq=True, metric='euclidean')[0][127,127])
        dtw_cost[i][j] = sum(costs)/len(costs)
key = 'dtw'
distances[key] = dtw_cost


print("Computed DTW cost.")

#Directed Hausdorff distance
hausdorff_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        hausdorff_distances[i][j] = (directed_hausdorff(all_flat[i], all_flat[j]))[0]
key = 'hau'
distances[key] = hausdorff_distances


print("Computed directed Hausdorff distances.")

#Minimum distance across all pairs
min_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        dists = []
        for n in range(kmax-kmin):
            for m in range(kmax-kmin):
                dists.append(np.linalg.norm(all_struct[i][n]-all_struct[j][m]))
        min_distances[i][j] = min(dists)
key = 'pair'
distances[key] = min_distances

print("Computed minimum pairwise distance.")


for f in range(0, file_no, 13):
    for metric in ['L1', 'fro', 'hau', 'pair', 'dtw']:
        D = np.zeros((13,13))
        for x in range(13):
            for y in range(13):
                D[x][y] = distances[metric][f+x][f+y]
        plt.figure()
        plt.matshow(D)
        plt.title(metric+'-'+all_names[f])
        plt.savefig(fig_dir+metric)


        
