'''
    Compute average distances for each metric on covers80
'''


def segment(y, s, rs_size, kmin, kmax, filter):
    """structurally segments the selected audio

        ds_size: side length to which combined matrix is going to be resampled to
        [kmin, kmax]: min and maximum approximation ranks
        filtering: True or False, whether memory stacking, timelag and path enhance are going to be used

    returns set of low rank approximations"""

    #compute cqt
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, 
                                        hop_length=512,
                                        bins_per_octave=12*3,
                                        n_bins=7*12*3)),
                                        ref=np.max)

    #beat synch cqt
    Csync = cv2.resize(C, (int(C.shape[1]/10), C.shape[0]))

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

    #downsample like CQT, compress time by 10
    Msync = cv2.resize(C, (int(mfcc.shape[1]/10), mfcc.shape[0]))

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
    
    # print("Cnorm shape:",Cnorm.shape)
    # plt.matshow(Cnorm)
    # plt.savefig(filedir[-10:-4])

    #approximations
    dist_set = []
    for k in range(kmin, kmax):

        # #debug
        # print(np.all(Cnorm[:, k-1:k]))
        # divisor = Cnorm[:, k-1:k]
        # if not np.all(divisor):
        #     print("0 divisor")

        Xs = evecs[:, :k] / Cnorm[:, k-1:k]
        

        #debug
        if np.isnan(np.sum(Xs)):
            print('woops')
            # fig, axs = plt.subplots(1, approx[1]-approx[0], figsize=(20, 20))
            # for i in range(approx[1]-approx[0]):
            #     axs[i].matshow(struct[i])
            # plt.savefig(filedir[-10:-1])

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
import csv

#--supress warnings--#
import warnings
warnings.filterwarnings("ignore")


#--reading--#

all_dirs = []
all_names = []
all_roots = []
all_audio = []
max_files = 40000
for root, dirs, files in os.walk('/home/ismir/Documents/ISMIR/Datasets/covers80/'):
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

#load audio
for f in range(file_no):
    y, sr = librosa.load(all_dirs[f], sr=16000, mono=True)
    #bug: empty mel bins
    all_audio.append((y,sr))

    #progress
    sys.stdout.write("\rLoading %i/%s pieces." % ((f+1), str(file_no)))
    sys.stdout.flush()
print('')


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
distances: L1, fro, dtw, hau, pair, sh2, sh3
format: rs_size-approx[0]-approx[1]-distance e.g. 128-2-8-L1
"""
distances = {}

#--Score dictionary--#
"""Terminology
distances: L1, fro, dtw, hau, pair, sh2, sh3
format: (filt-)rs_size-approx[0]-approx[1]-distance e.g. filt-128-2-8-L1
"""
scores = {}


#--traverse parameters, compute segmentations, save evaluation--#

#resampling parameters
#for rs_size in [32]:
for rs_size in [128]:
    #approximations
    #for approx in [[2,6]]:
    for approx in [[2,11]]:
        for filtering in [True]:

            #string for keys to indicate filtering
            if filtering:
                filt = 'filt-'
            else:
                filt = ''

            #hold all structures and their formats
            all_struct = [] #kmax-kmin sets each with a square matrix
            all_flat = [] #kmax-kmin sets each with a flattened matrix
            all_merged = [] #single concatenated vector with all flattened matrices
            all_shingled2 = [] #shingled pairs of flat approximations
            all_shingled3 = [] #shingled triples of flat approximations

            print("--------------------")
            print("Resampling size:", str(rs_size))
            print("Approximation range: [" + str(approx[0]) + ',' + str(approx[1]) + ']')
            print("Filtering:", str(filtering))

            #songs
            for f in range(file_no):
                #structure segmentation
                struct = segment(all_audio[f][0], all_audio[f][1],
                                rs_size, approx[0], approx[1], filtering)
                all_struct.append(struct)

                # #debug
                # fig, axs = plt.subplots(1, approx[1]-approx[0], figsize=(20, 20))
                # for i in range(approx[1]-approx[0]):
                #     axs[i].matshow(struct[i])
                # plt.savefig(all_names[f])

                #formatting
                flat_approximations = []
                merged_approximations = np.empty((0))
                for j in range(approx[1]-approx[0]):
                    flat_approximations.append(struct[j].flatten())
                    merged_approximations = np.concatenate((merged_approximations, flat_approximations[j]))
                all_flat.append(np.asarray(flat_approximations))
                all_merged.append(merged_approximations)

                #shingling per 2
                shingled = []
                for j in range(approx[1]-approx[0]-1):
                    shingled.append(np.concatenate((all_flat[f][j],all_flat[f][j+1]),axis=None))
                    #shingled.append(np.concatenate((struct[all_names[f]]['OG'][1][j],struct[all_names[f]]['OG'][1][j+1]),axis=None))
                all_shingled2.append(np.asarray(shingled))

                #shingling per 3
                shingled = []
                for j in range(approx[1]-approx[0]-2):
                    shingled.append(np.concatenate((all_flat[f][j],all_flat[f][j+1],all_flat[f][j+2]),axis=None))
                    #shingled.append(np.concatenate((struct[all_names[f]]['OG'][1][j],struct[all_names[f]]['OG'][1][j+1],struct[all_names[f]]['OG'][1][j+2]), axis=None))
                all_shingled3.append(np.asarray(shingled))
                
                #progress
                sys.stdout.write("\rSegmented %i/%s pieces." % ((f+1), str(file_no)))
                sys.stdout.flush()
            print('')

            # #plot approximations
            # fig, axs = plt.subplots(1, approx[1]-approx[0], figsize=(20, 20))
            # for i in range(approx[1]-approx[0]):
            #     axs[i].matshow(all_struct[0][i])
            # plt.savefig('approximations'+str(rs_size))

            #list to numpy array
            all_struct = np.asarray(all_struct)
            all_flat = np.asarray(all_flat)
            all_merged = np.asarray(all_merged)

            rows = [['', 'mean', 'max']]

            #L1 norm
            L1_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    L1_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j], ord=1)

            rows.append(['L1', np.mean(L1_distances), np.amax(L1_distances)])

            #Frobenius norm
            fro_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    fro_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j])

            rows.append(['Frobenius', np.mean(fro_distances), np.amax(fro_distances)])

            #Sub-sequence Dynamic Time Warping cost
            dtw_cost = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    costs = []
                    for k in range(approx[1]-approx[0]):           
                        costs.append(librosa.sequence.dtw(all_struct[i][k], all_struct[j][k], subseq=False, metric='euclidean')[0][rs_size-1,rs_size-1])
                    dtw_cost[i][j] = sum(costs)/len(costs)

            rows.append(['DTW', np.mean(dtw_cost), np.amax(dtw_cost)])
            
            #Directed Hausdorff distance
            hausdorff_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    hausdorff_distances[i][j] = (directed_hausdorff(all_flat[i], all_flat[j]))[0]

            rows.append(['Hausdorff', np.mean(hausdorff_distances), np.amax(hausdorff_distances)])
            
            #Minimum distance across all pairs
            min_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    dists = []
                    for n in range(approx[1]-approx[0]):
                        for m in range(approx[1]-approx[0]):
                            dists.append(np.linalg.norm(all_struct[i][n]-all_struct[j][m]))
                    min_distances[i][j] = min(dists)
            
            rows.append(['Pair', np.mean(min_distances), np.amax(min_distances)])

            #Directed Hausdorff distance of shingled pairs
            shingled2_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    shingled2_distances[i][j] = (directed_hausdorff(all_shingled2[i], all_shingled2[j]))[0]

            rows.append(['Shingled 2', np.mean(shingled2_distances), np.amax(shingled2_distances)])

            #Directed Hausdorff distance of shingled triples
            shingled3_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    shingled3_distances[i][j] = (directed_hausdorff(all_shingled3[i], all_shingled3[j]))[0]

            rows.append(['Shingled 3', np.mean(shingled3_distances), np.amax(shingled3_distances)])

with open('/home/ismir/Documents/ISMIR/figures/deformations_run2/mean_max.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
print('Stats computed.')