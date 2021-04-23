def segment(filedir, rs_size, kmin, kmax):
    """structurally segments the selected audio

        ds_size: side length to which combined matrix is going to be resampled to
        [kmin, kmax]: min and maximum approximation ranks

    returns set of low rank approximations"""

    #load audio
    y, sr = librosa.load(filedir, sr=16000, mono=True)

    #compute cqt
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                        bins_per_octave=12*3,
                                        n_bins=7*12*3)),
                                        ref=np.max)

    #beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)

    #beat synch cqt
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    #stack memory
    Cstack = librosa.feature.stack_memory(Csync, 4)

    #Affinity matrix
    R = librosa.segment.recurrence_matrix(Cstack, width=3, mode='affinity', sym=True)

    #Filtering
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))
    Rf = librosa.segment.path_enhance(Rf, 15)

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
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path

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
    dist_set = []
    for k in range(kmin, kmax):
        Xs = evecs[:, :k] / Cnorm[:, k-1:k]
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
for root, dirs, files in os.walk('/Users/chris/Google Drive/Classes/Capstone/Datasets/deformations/'):
        for name in files:
            if (('.wav' in name) or ('.aif' in name) or ('.mp3' in name)):
                filepath = os.path.join(root, name)
                all_dirs.append(filepath)
                all_names.append(name[:-4])
                all_roots.append(root)
                if len(all_dirs)>=max_files:
                    break
        if len(all_dirs)>=max_files:
            break        
file_no = len(all_dirs)


#--same song (True) vs different song (False)--#
covers = np.zeros((file_no, file_no), dtype=np.bool_)
for i in range(file_no):
    for j in range(file_no):
        if (all_roots[i] == all_roots[j]):
            covers[i][j] = True
        else:
            covers[i][j] = False

#--Distance dictionary--#
#L1, fro, hau, dtw, pair
distances = {}


#--traverse parameters, compute segmentations, save evaluation--#

all_struct = [] #kmax-kmin sets each with a square matrix
all_flat = [] #kmax-kmin sets each with a flattened matrix
all_merged = [] #single concatenated vector with all flattened matrices


#songs
for f in range(file_no):
    #structure segmentation
    struct = segment(all_dirs[f], 128, 2, 8)
    all_struct.append(struct)

    #formatting
    flat_approximations = []
    merged_approximations = np.empty((0))
    for j in range(6):
        flat_approximations.append(struct[j].flatten())
        merged_approximations = np.concatenate((merged_approximations, flat_approximations[j]))
    all_flat.append(np.asarray(flat_approximations))
    all_merged.append(merged_approximations)
    
    #progress
    sys.stdout.write("\rSegmented %i/%s pieces." % ((f+1), str(file_no)))
    sys.stdout.flush()
print('')

# #plot approximations
# fig, axs = plt.subplots(1, 6, figsize=(20, 20))
# for i in range(6):
#     axs[i].matshow(all_struct[0][i])
# plt.savefig('approximations')

#list to numpy array
all_struct = np.asarray(all_struct)
all_flat = np.asarray(all_flat)
all_merged = np.asarray(all_merged)

print(np.sum(all_merged))

#figure directory
fig_dir = '/Users/chris/Google Drive/Classes/Capstone/figures/deformations/'

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
        for k in range(6):           
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
        for n in range(6):
            for m in range(6):
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


        
