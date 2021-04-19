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
max_files = 10
for root, dirs, files in os.walk('/Users/chris/Google Drive/Classes/Capstone/Datasets/covers80/covers32k'):
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


#--cover (True) vs non-cover (False)--#
covers = np.zeros((file_no, file_no), dtype=np.bool_)
for i in range(file_no):
    for j in range(file_no):
        if (all_roots[i] == all_roots[j]):
            covers[i][j] = True
        else:
            covers[i][j] = False

#--Distance dictionary--#
"""Terminology
distances: L1, fro, dtw, hau, pair
format: rs_size-approx[0]-approx[1]-distance e.g. 128-2-8-L1
"""
distances = {}


#--traverse parameters, compute segmentations, save evaluation--#

all_struct = [] #kmax-kmin sets each with a square matrix
all_flat = [] #kmax-kmin sets each with a flattened matrix
all_merged = [] #single concatenated vector with all flattened matrices

#resampling parameters
for rs_size in [32, 64, 128, 256]:
    #approximations
    for approx in [[2,6], [2,10], [3,7], [3,11]]:

        print("--------------------")
        print("Resampling size:", str(rs_size))
        print("Approximation range: [" + str(approx[0]) + ',' + str(approx[1]) + ']')

        #songs
        for f in range(file_no):
            #structure segmentation
            struct = segment(all_dirs[f], rs_size, approx[0], approx[1])
            all_struct.append(struct)

            #formatting
            flat_approximations = []
            merged_approximations = np.empty((0))
            for j in range(approx[1]-approx[0]):
                flat_approximations.append(struct[j].flatten())
                merged_approximations = np.concatenate((merged_approximations, flat_approximations[j]))
            all_flat.append(np.asarray(flat_approximations))
            all_merged.append(merged_approximations)
            
            #progress
            sys.stdout.write("\rSegmented %i/%s pieces." % ((f+1), str(file_no)))
            sys.stdout.flush()
        print('')

        #plot approximations
        fig, axs = plt.subplots(1, approx[1]-approx[0], figsize=(20, 20))
        for i in range(approx[1]-approx[0]):
            axs[i].matshow(all_struct[0][i])
        plt.savefig('approximations'+str(rs_size))

        #list to numpy array
        all_struct = np.asarray(all_struct)
        all_flat = np.asarray(all_flat)
        all_merged = np.asarray(all_merged)

        #figure directory
        fig_dir = '../../../figures/covers80/'

        #L1 norm
        L1_distances = np.zeros((file_no, file_no))
        for i in range(file_no):
            for j in range(file_no):
                L1_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j], ord=1)

        key = str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1])+'-L1'
        distances[key] = L1_distances
        
        L1_distances_covers = []
        L1_distances_noncovers = []
        for i in range(file_no):
            for j in range(file_no):
                if covers[i][j]:
                    if (L1_distances[i][j] != 0):
                        L1_distances_covers.append(L1_distances[i][j])
                else:
                    L1_distances_noncovers.append(L1_distances[i][j])
        plt.figure()
        plt.hist(L1_distances_covers, bins=100, alpha=0.5, label='Covers', density=1)
        plt.hist(L1_distances_noncovers, bins=100, alpha=0.5, label='Non-covers', density=1)
        plt.title("Histogram of L1 distances between cover and non-cover pairs")
        plt.legend(loc='upper right')
        plt.savefig(fig_dir+key+'-hist')

        hit_positions = []
        for i in range(file_no):
            for cover_idx in range(file_no):
                if covers[i][cover_idx] and i!=cover_idx:
                    j = cover_idx
            d = L1_distances[i]
            d = np.argsort(d)
            hit = np.where(d==j)[0][0]
            hit_positions.append(hit)
        plt.figure()
        plt.plot(hit_positions)
        plt.title('Position of hit - Average: ' + str(np.mean(hit_positions)))
        plt.savefig(fig_dir+key+'-hit_pos')

        print("Computed L1 distances.")

        #Frobenius norm
        fro_distances = np.zeros((file_no, file_no))
        for i in range(file_no):
            for j in range(file_no):
                fro_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j])
        key = str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1])+'-fro'
        distances[key] = fro_distances

        fro_distances_covers = []
        fro_distances_noncovers = []
        for i in range(file_no):
            for j in range(file_no):
                if covers[i][j]:
                    if (fro_distances[i][j] != 0):
                        fro_distances_covers.append(fro_distances[i][j])
                else:
                    fro_distances_noncovers.append(fro_distances[i][j])         
        plt.figure()
        plt.hist(fro_distances_covers, bins=100, alpha=0.5, label='Covers', density=1)
        plt.hist(fro_distances_noncovers, bins=100, alpha=0.5, label='Non-covers', density=1)
        plt.title("Histogram of Frobenius distances between cover and non-cover pairs")
        plt.legend(loc='upper right')
        plt.savefig(fig_dir+key+'-hist')

        hit_positions = []
        for i in range(file_no):
            for cover_idx in range(file_no):
                if covers[i][cover_idx] and i!=cover_idx:
                    j = cover_idx
            d = fro_distances[i]
            d = np.argsort(d)
            hit = np.where(d==j)[0][0]
            hit_positions.append(hit)
        plt.figure()
        plt.plot(hit_positions)
        plt.title('Position of hit - Average: ' + str(np.mean(hit_positions)))
        plt.savefig(fig_dir+key+'-hit_pos')

        print("Computed Frobenius distances.")

        #Sub-sequence Dynamic Time Warping cost
        dtw_cost = np.zeros((file_no, file_no))
        for i in range(file_no):
            for j in range(file_no):
                costs = []
                for k in range(approx[1]-approx[0]):           
                    costs.append(librosa.sequence.dtw(all_struct[i][k], all_struct[j][k], subseq=True, metric='euclidean')[0][rs_size-1,rs_size-1])
                dtw_cost[i][j] = sum(costs)/len(costs)
        key = str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1])+'-dtw'
        distances[key] = dtw_cost

        dtw_cost_covers = []
        dtw_cost_noncovers = []
        for i in range(file_no):
            for j in range(file_no):
                if covers[i][j]:
                    if (dtw_cost[i][j] != 0):
                        dtw_cost_covers.append(dtw_cost[i][j])
                else:
                    dtw_cost_noncovers.append(dtw_cost[i][j]) 
        plt.figure()
        plt.hist(dtw_cost_covers, bins=100, alpha=0.5, label='Covers', density=1)
        plt.hist(dtw_cost_noncovers, bins=100, alpha=0.5, label='Non-covers', density=1)
        plt.title("Histogram of subsequence DTW cost between cover and non-cover pairs")
        plt.legend(loc='upper right')
        plt.savefig(fig_dir+key+'-hist')

        hit_positions = []
        for i in range(file_no):
            for cover_idx in range(file_no):
                if covers[i][cover_idx] and i!=cover_idx:
                    j = cover_idx
            d = dtw_cost[i]
            d = np.argsort(d)
            hit = np.where(d==j)[0][0]
            hit_positions.append(hit)
        plt.figure()
        plt.plot(hit_positions)
        plt.title('Position of hit - Average: ' + str(np.mean(hit_positions)))
        plt.savefig(fig_dir+key+'-hit_pos')

        print("Computed DTW cost.")

        #Directed Hausdorff distance
        hausdorff_distances = np.zeros((file_no, file_no))
        for i in range(file_no):
            for j in range(file_no):
                hausdorff_distances[i][j] = (directed_hausdorff(all_flat[i], all_flat[j]))[0]
        key = str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1])+'-hau'
        distances[key] = hausdorff_distances

        hausdorff_distances_covers = []
        hausdorff_distances_noncovers = []
        for i in range(file_no):
            for j in range(file_no):
                if covers[i][j]:
                    if (hausdorff_distances[i][j] != 0):
                        hausdorff_distances_covers.append(hausdorff_distances[i][j])
                else:
                    hausdorff_distances_noncovers.append(hausdorff_distances[i][j])             
        plt.figure()
        plt.hist(hausdorff_distances_covers, bins=100, alpha=0.5, label='Covers', density=1)
        plt.hist(hausdorff_distances_noncovers, bins=100, alpha=0.5, label='Non-covers', density=1)
        plt.title("Histogram of Hausdorff distances between cover and non-cover pairs")
        plt.legend(loc='upper right')
        plt.savefig(fig_dir+key+'-hist')

        hit_positions = []
        for i in range(file_no):
            for cover_idx in range(file_no):
                if covers[i][cover_idx] and i!=cover_idx:
                    j = cover_idx
            d = hausdorff_distances[i]
            d = np.argsort(d)
            hit = np.where(d==j)[0][0]
            hit_positions.append(hit)
        plt.figure()
        plt.plot(hit_positions)
        plt.title('Position of hit - Average: ' + str(np.mean(hit_positions)))
        plt.savefig(fig_dir+key+'-hit_pos')

        print("Computed directed Hausdorff distances.")

        #Minimum distance across all pairs
        min_distances = np.zeros((file_no, file_no))
        for i in range(file_no):
            for j in range(file_no):
                dists = []
                for n in range(approx[1]-approx[0]):
                    for m in range(approx[1]-approx[0]):
                        dists.append(np.linalg.norm(all_struct[i][n]-all_struct[j][m]))
                min_distances[i][j] = min(dists)
        key = str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1])+'-pair'
        distances[key] = min_distances

        min_distances_covers = []
        min_distances_noncovers = []
        for i in range(file_no):
            for j in range(file_no):
                if covers[i][j]:
                    if (min_distances[i][j] != 0):
                        min_distances_covers.append(min_distances[i][j])
                else:
                    min_distances_noncovers.append(min_distances[i][j])            
        plt.figure()
        plt.hist(min_distances_covers, bins=100, alpha=0.5, label='Covers', density=1)
        plt.hist(min_distances_noncovers, bins=100, alpha=0.5, label='Non-covers', density=1)
        plt.title("Histogram of min pair distances between cover and non-cover pairs")
        plt.legend(loc='upper right')
        plt.savefig(fig_dir+key+'-hist')

        hit_positions = []
        for i in range(file_no):
            for cover_idx in range(file_no):
                if covers[i][cover_idx] and i!=cover_idx:
                    j = cover_idx
            d = min_distances[i]
            d = np.argsort(d)
            hit = np.where(d==j)[0][0]
            hit_positions.append(hit)
        plt.figure()
        plt.plot(hit_positions)
        plt.title('Position of hit - Average: ' + str(np.mean(hit_positions)))
        plt.savefig(fig_dir+key+'-hit_pos')

        print("Computed minimum pairwise distance.")

