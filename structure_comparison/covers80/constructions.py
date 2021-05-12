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
import dill
import sys
import glob
import os
import random

#change matplotlib backend to save rendered plots correctly on linux 
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

#--supress warnings--#
import warnings
warnings.filterwarnings("ignore")


#-------------#
#---reading---#
#-------------#
all_dirs = []
all_names = []
all_roots = []
all_audio = []
max_files = 5
for root, dirs, files in os.walk('/home/chris/Documents/datasets/covers80/covers32k/'):
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

#----------------#
#---load audio---#
#----------------#
for f in range(file_no):
    y, sr = librosa.load(all_dirs[f], sr=16000, mono=True)
    #bug: empty mel bins
    all_audio.append((y,sr))

    #progress
    sys.stdout.write("\rLoading %i/%s pieces." % ((f+1), str(file_no)))
    sys.stdout.flush()
print('')

#-----------#
#---cover---#
#-----------#  cover (True) vs non-cover (False)
covers = np.zeros((file_no, file_no), dtype=np.bool_)
for i in range(file_no):
    for j in range(file_no):
        if (all_roots[i] == all_roots[j]):
            covers[i][j] = True
        else:
            covers[i][j] = False

#-------------------------#
#---Distance dictionary---#
#-------------------------#
"""Terminology
distances: L1, fro, dtw, hau, pair, sh2, sh3
format: rs_size-approx[0]-approx[1]-distance e.g. 128-2-8-L1
"""
distances = {}

#----------------------#
#---Score dictionary---#
#----------------------#
"""Terminology
distances: L1, fro, dtw, hau, pair, sh2, sh3
format: (filt-)rs_size-approx[0]-approx[1]-distance e.g. filt-128-2-8-L1
"""
scores = {}


#--------------------------------------------#
#---traverse parameters, segment, evaluate---#
#--------------------------------------------#

#figure directory
fig_dir = '/home/chris/Documents/figures/'

#for rs_size in [32]: #test config
for rs_size in [32, 64, 128, 256]: #resampling parameters
    #for approx in [[2,6]]: #test config
    for approx in [[2,6], [2,8], [4,10], [8,12]]: #min number of approximations is 3
        for filtering in [True, False]:

            #string for keys to indicate filtering
            if filtering:
                filt = 'filt-'
            else:
                filt = ''

            #print configuration
            print("--------------------")
            print("Resampling size:", str(rs_size))
            print("Approximation range: [" + str(approx[0]) + ',' + str(approx[1]) + ']')
            print("Filtering:", str(filtering))



            #--------------------#
            #--------------------#
            #-----Formatting-----#
            #--------------------#
            #--------------------#

            #hold all structures and their formats
            all_struct = [] #kmax-kmin sets each with a square matrix
            all_flat = [] #kmax-kmin sets each with a flattened matrix
            all_merged = [] #single concatenated vector with all flattened matrices
            all_shingled2 = [] #shingle adjacent pairs of flat approoximations
            all_shingled3 = [] #shingle adjacent triples of flat approoximations

            #traverse songs
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
                    #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1]]))
                    shingled.append(np.concatenate((all_flat[f][j],all_flat[f][j+1]), axis=None))
                all_shingled2.append(np.asarray(shingled))

                #shingling per 3
                shingled = []
                for j in range(approx[1]-approx[0]-2):
                    #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1],all_flat[f][j+2]]))
                    shingled.append(np.concatenate((all_flat[f][j],all_flat[f][j+1],all_flat[f][j+2]), axis=None))
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
            all_shingled2 = np.asarray(all_shingled2)
            all_shingled3 = np.asarray(all_shingled3)


            #-------------------#
            #-------------------#
            #-----Distances-----#
            #-------------------#
            #-------------------#

            #---------#
            #-L1 norm-#
            #---------#

            L1_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    L1_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j], ord=1)

            key = filt + str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1]-1)+'-L1'
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
            hit_mean = np.mean(hit_positions)
            plt.title('Position of hit - Average: ' + str(hit_mean))
            plt.savefig(fig_dir+key+'-hit_pos')
            scores[key]=hit_mean

            print("Computed L1 distances.")


            #----------------#
            #-Frobenius norm-#
            #----------------#

            fro_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    fro_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j])
            key = filt + str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1]-1)+'-fro'
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
            hit_mean = np.mean(hit_positions)
            plt.title('Position of hit - Average: ' + str(hit_mean))
            plt.savefig(fig_dir+key+'-hit_pos')
            scores[key]=hit_mean

            print("Computed Frobenius distances.")


            #----------------------------------------#
            #-Sub-sequence Dynamic Time Warping cost-#
            #----------------------------------------#

            dtw_cost = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    costs = []
                    for k in range(approx[1]-approx[0]):           
                        costs.append(librosa.sequence.dtw(all_struct[i][k], all_struct[j][k], subseq=False, metric='euclidean')[0][rs_size-1,rs_size-1])
                    dtw_cost[i][j] = sum(costs)/len(costs)
            key = filt + str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1]-1)+'-dtw'
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
            hit_mean = np.mean(hit_positions)
            plt.title('Position of hit - Average: ' + str(hit_mean))
            plt.savefig(fig_dir+key+'-hit_pos')
            scores[key] = hit_mean

            print("Computed DTW cost.")


            #-----------------------------#
            #-Directed Hausdorff distance-#
            #-----------------------------#

            hausdorff_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    hausdorff_distances[i][j] = (directed_hausdorff(all_flat[i], all_flat[j]))[0]
            key = filt + str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1]-1)+'-hau'
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
            hit_mean = np.mean(hit_positions)
            plt.title('Position of hit - Average: ' + str(hit_mean))
            plt.savefig(fig_dir+key+'-hit_pos')
            scores[key] = hit_mean

            print("Computed directed Hausdorff distances.")


            #-----------------------------------#
            #-Minimum distance across all pairs-#
            #-----------------------------------#

            min_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    dists = []
                    for n in range(approx[1]-approx[0]):
                        for m in range(approx[1]-approx[0]):
                            dists.append(np.linalg.norm(all_struct[i][n]-all_struct[j][m]))
                    min_distances[i][j] = min(dists)
            key = filt + str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1]-1)+'-pair'
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
            hit_mean = np.mean(hit_positions)
            plt.title('Position of hit - Average: ' + str(hit_mean))
            plt.savefig(fig_dir+key+'-hit_pos')
            scores[key] = hit_mean

            print("Computed minimum pairwise distance.")


            #---------------------------------------------#
            #-Directed Hausdorff distance shingled tuples-#
            #---------------------------------------------#

            shingled2_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    shingled2_distances[i][j] = (directed_hausdorff(all_shingled2[i], all_shingled2[j]))[0]
            key = filt + str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1]-1)+'-sh2'
            distances[key] = shingled2_distances

            shingled2_distances_covers = []
            shingled2_distances_noncovers = []
            for i in range(file_no):
                for j in range(file_no):
                    if covers[i][j]:
                        if (shingled2_distances[i][j] != 0):
                            shingled2_distances_covers.append(shingled2_distances[i][j])
                    else:
                        shingled2_distances_noncovers.append(shingled2_distances[i][j])             
            plt.figure()
            plt.hist(shingled2_distances_covers, bins=100, alpha=0.5, label='Covers', density=1)
            plt.hist(shingled2_distances_noncovers, bins=100, alpha=0.5, label='Non-covers', density=1)
            plt.title("Histogram of Hausdorff distances between cover and non-cover pairs of shingled tuples")
            plt.legend(loc='upper right')
            plt.savefig(fig_dir+key+'-hist')

            hit_positions = []
            for i in range(file_no):
                for cover_idx in range(file_no):
                    if covers[i][cover_idx] and i!=cover_idx:
                        j = cover_idx
                d = shingled2_distances[i]
                d = np.argsort(d)
                hit = np.where(d==j)[0][0]
                hit_positions.append(hit)
            plt.figure()
            plt.plot(hit_positions)
            hit_mean = np.mean(hit_positions)
            plt.title('Position of hit - Average: ' + str(hit_mean))
            plt.savefig(fig_dir+key+'-hit_pos')
            scores[key] = hit_mean

            print("Computed directed Hausdorff distances f bigrams.")


            #-------------------------------------#
            #-Directed Hausdorff shingled triples-#
            #-------------------------------------#

            shingled3_distances = np.zeros((file_no, file_no))
            for i in range(file_no):
                for j in range(file_no):
                    shingled3_distances[i][j] = (directed_hausdorff(all_flat[i], all_flat[j]))[0]
            key = filt + str(rs_size)+'-'+str(approx[0])+'-'+str(approx[1]-1)+'-sh3'
            distances[key] = shingled3_distances

            shingled3_distances_covers = []
            shingled3_distances_noncovers = []
            for i in range(file_no):
                for j in range(file_no):
                    if covers[i][j]:
                        if (shingled3_distances[i][j] != 0):
                            shingled3_distances_covers.append(shingled3_distances[i][j])
                    else:
                        shingled3_distances_noncovers.append(shingled3_distances[i][j])             
            plt.figure()
            plt.hist(shingled3_distances_covers, bins=100, alpha=0.5, label='Covers', density=1)
            plt.hist(shingled3_distances_noncovers, bins=100, alpha=0.5, label='Non-covers', density=1)
            plt.title("Histogram of Hausdorff distances between cover and non-cover pairs of shingled triples")
            plt.legend(loc='upper right')
            plt.savefig(fig_dir+key+'-hist')

            hit_positions = []
            for i in range(file_no):
                for cover_idx in range(file_no):
                    if covers[i][cover_idx] and i!=cover_idx:
                        j = cover_idx
                d = shingled3_distances[i]
                d = np.argsort(d)
                hit = np.where(d==j)[0][0]
                hit_positions.append(hit)
            plt.figure()
            plt.plot(hit_positions)
            hit_mean = np.mean(hit_positions)
            plt.title('Position of hit - Average: ' + str(hit_mean))
            plt.savefig(fig_dir+key+'-hit_pos')
            scores[key] = hit_mean

            print("Computed directed Hausdorff distances of trigrams.")

dill.dump_session('/home/chris/Documents/dills/5covers80_all.db')