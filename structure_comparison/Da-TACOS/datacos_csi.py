# Script for Da-Tacos cover song identification from Feature Fused Matrices

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
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import cv2
from sklearn import metrics
import dill
import sys
import glob
import os
import random
import json
import deepdish as dd

#change matplotlib backend to save rendered plots correctly on linux 
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

# #--supress warnings--#
# import warnings
# warnings.filterwarnings("ignore")


#---Load metadata---#
with open('/home/ismir/Documents/ISMIR/Datasets/da-tacos/da-tacos_benchmark_subset_metadata.json') as f:
    benchmark_metadata = json.load(f)

#---Segmentation parameters---#
rs_size = 128
kmin = 8
kmax = 12

#---Counters---#
count = 0
W_count=0
P_count = 0

#---Loading limits---#
min_covers = 10 #load works for which there are at least min_covers performances
max_covers = 10 #stop loading performances if over max_covers per work
max_works = 100

#---Storage---#
all_sets = []
#all_shapeDNAs = []
all_WP = []
y = []

#for all Works
for W in benchmark_metadata.keys():
    if len(benchmark_metadata[W].keys()) >= min_covers: #if it contains at least 5 covers
        P_count = 0
        #for all performances
        for P in benchmark_metadata[W].keys():
            P_count += 1
            
            #Computations
            try:
                SSM = dd.io.load("/home/ismir/Documents/ISMIR/Datasets/da-tacosSSMs/StructureLaplacian_datacos_crema_" + P + ".h5")['WFused']
            except:
                print("Couldn't load " + P + ".")
                continue

            N = dd.io.load("/home/ismir/Documents/ISMIR/Datasets/da-tacosSSMs/StructureLaplacian_datacos_crema_" + P + ".h5")['N']

            #Construct square matrix from flattened upper triangle
            A = np.zeros((N,N))
            iN = np.triu_indices(N) #return indices for upper-triangle of (N,N) matrix
            for i in range(len(SSM)):
                A[iN[0][i]][iN[1][i]] = SSM[i]
            B = np.transpose(A)
            square_SSM = A+B

            #Resample
            SSM_ds = cv2.resize(square_SSM, (rs_size,rs_size))

            #Compute the Laplacian
            L = laplacian(SSM_ds, normed=True)

            #Laplacian eigenvalues and eigenvectors
            evals, evecs = eigh(L)

            # #Shape DNA
            # shapeDNA = evals[:30]
            # all_shapeDNAs.append(shapeDNA)

            #Hierarchical structure
            evecs = median_filter(evecs, size=(9, 1))
            Cnorm = np.cumsum(evecs**2, axis=1)**0.5
            # #temporary replacement for bug
            # a_min_value = 3.6934424e-08
            # Cnorm[Cnorm == 0.0] = a_min_value
            # if (np.isnan(np.sum(Cnorm))):
            #     print("WOOOOOAH")
            
            dist_set = []
            for k in range(kmin, kmax):
                X = evecs[:, :k] / Cnorm[:, k-1:k]
                distance = squareform(pdist(X, metric='euclidean'))
                dist_set.append(distance)

            all_sets.append(dist_set)
            y.append(W)

            #append W and P
            all_WP.append([W, P])

            #plt.matshow()
            #plt.colorbar()
            #plt.show()

            if (P_count >=max_covers):
                break
                
        W_count +=1
        sys.stdout.write("\rLoading %i works." % W_count)
        sys.stdout.flush()
        if (W_count >= max_works):
            break
            
all_sets = np.asarray(all_sets)
file_no = len(all_WP)
# all_shapeDNAs = np.asarray(all_shapeDNAs)

print("\nLoaded Da-TACOS SMMs.")
print("Data shape:", all_sets.shape)

fig, axs = plt.subplots(1, kmax-kmin, figsize=(20, 20))
for i in range(kmax-kmin):
    axs[i].matshow(all_sets[8][i])
plt.savefig('/home/ismir/Documents/ISMIR/figures/datacos/approx.png')

#------------#
#-Formatting-#
#------------#

all_flat = [] #kmin-kmin sets each with a flattened matrix
all_merged = [] #single concatenated vector with all flattened matrices
all_shingled2 = [] #shingle adjacent pairs of flat approoximations
all_shingled3 = [] #shingle adjacent triples of flat approoximations

#traverse songs
for f in range(file_no):

    #formatting
    flat_approximations = []
    merged_approximations = np.empty((0))
    for j in range(kmax-kmin):
        flat_approximations.append(all_sets[f][j].flatten())
        merged_approximations = np.concatenate((merged_approximations, flat_approximations[j]))
    all_flat.append(np.asarray(flat_approximations))
    all_merged.append(merged_approximations)

    #shingling per 2
    shingled = []
    for j in range(kmax-kmin-1):
        #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1]]))
        shingled.append(np.concatenate((all_flat[f][j],all_flat[f][j+1]), axis=None))
    all_shingled2.append(np.asarray(shingled))

    #shingling per 3
    shingled = []
    for j in range(kmax-kmin-2):
        #shingled.append(np.array([all_flat[f][j],all_flat[f][j+1],all_flat[f][j+2]]))
        shingled.append(np.concatenate((all_flat[f][j],all_flat[f][j+1],all_flat[f][j+2]), axis=None))
    all_shingled3.append(np.asarray(shingled))

    #progress
    sys.stdout.write("\rFormatted %i/%s approximation sets." % ((f+1), str(file_no)))
    sys.stdout.flush()
print('')

all_flat = np.asarray(all_flat)
all_merged = np.asarray(all_merged)
all_shingled2 = np.asarray(all_shingled2)
all_shingled3 = np.asarray(all_shingled3)

#----------------------#
#-Covers vs Non-covers-#
#----------------------#

#True if cover, False if non-cover
covers = np.zeros((len(all_WP), len(all_WP)), dtype=np.bool_)
for i in range(len(all_WP)):
    for j in range(len(all_WP)):
        if (all_WP[i][0] == all_WP[j][0]):
            covers[i][j] = True
        else:
            covers[i][j] = False

#-----------#
#-Distances-#
#-----------#

fig_dir = '/home/ismir/Documents/ISMIR/figures/datacos/'

#---L1---#
L1_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        L1_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j], ord=1)

#Histogram
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
plt.hist(L1_distances_covers, bins=200, alpha=0.5, label='Covers', density=1)
plt.hist(L1_distances_noncovers, bins=200, alpha=0.5, label='Non-covers', density=1)
plt.title("Histogram of L1 distances between cover and non-cover pairs")
plt.legend(loc='upper right')
plt.savefig(fig_dir+'Histogram-L1norm.png')

#Mean position of first hit
hit_positions = []
for i in range(file_no):
    cvrs = [] #list of cover indeces for that work
    for cover_idx in range(file_no):
        if covers[i][cover_idx] and i!=cover_idx: #if cover and not the same work
            cvrs.append(cover_idx)
    d = L1_distances[i]
    d = np.argsort(d)
    hits = []
    for c in range(len(cvrs)): #traverse covers
        hits.append(np.where(d==c)[0][0])
    hit_positions.append(min(hits))
L1_average_hit = np.mean(hit_positions)
print('L1 mean position of first hit:', L1_average_hit)

#Mean Average Precision
for i in range(file_no):
    #get all distances to selected song, normalize [0,1], convert to similarity metric, not dissimilarity
    d = 1-(L1_distances[i]/np.linalg.norm(L1_distances[i])) 
    c = covers[1] #get all cover relationships to selected song
    mAP = 0
    for j in range(file_no):
        mAP += average_precision_score(c, d)
    mAP = mAP/float(file_no)
print('L1 mean average precision:', mAP)


#---Frobenius norm---#
fro_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        fro_distances[i][j] = np.linalg.norm(all_merged[i]-all_merged[j])

#Histogram
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
plt.hist(fro_distances_covers, bins=200, alpha=0.5, label='Covers', density=1)
plt.hist(fro_distances_noncovers, bins=200, alpha=0.5, label='Non-covers', density=1)
plt.title("Histogram of fro distances between cover and non-cover pairs")
plt.legend(loc='upper right')
plt.savefig(fig_dir+'Histogram-fronorm.png')

#Mean position of first hit
hit_positions = []
for i in range(file_no):
    cvrs = [] #list of cover indeces for that work
    for cover_idx in range(file_no):
        if covers[i][cover_idx] and i!=cover_idx: #if cover and not the same work
            cvrs.append(cover_idx)
    d = fro_distances[i]
    d = np.argsort(d)
    hits = []
    for c in range(len(cvrs)): #traverse covers
        hits.append(np.where(d==c)[0][0])
    hit_positions.append(min(hits))
fro_average_hit = np.mean(hit_positions)
print('fro mean position of first hit:', fro_average_hit)

#Mean Average Precision
for i in range(file_no):
    #get all distances to selected song, normalize [0,1], convert to similarity metric, not dissimilarity
    d = 1-(fro_distances[i]/np.linalg.norm(fro_distances[i])) 
    c = covers[1] #get all cover relationships to selected song
    mAP = 0
    for j in range(file_no):
        mAP += average_precision_score(c, d)
    mAP = mAP/float(file_no)
print('fro mean average precision:', mAP)


#---Sub-sequence Dynamic Time Warping Cost---#

dtw_cost = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        costs = []
        for k in range(kmax-kmin):           
            costs.append(librosa.sequence.dtw(all_sets[i][k], all_sets[j][k], subseq=False, metric='euclidean')[0][rs_size-1,rs_size-1])
        dtw_cost[i][j] = sum(costs)/len(costs)

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
plt.hist(dtw_cost_covers, bins=200, alpha=0.5, label='Covers', density=1)
plt.hist(dtw_cost_noncovers, bins=200, alpha=0.5, label='Non-covers', density=1)
plt.title("Histogram of subsequence DTW cost between cover and non-cover pairs")
plt.legend(loc='upper right')
plt.savefig(fig_dir+'Histogram-dtw.png')

#Mean position of first hit
hit_positions = []
for i in range(file_no):
    cvrs = [] #list of cover indeces for that work
    for cover_idx in range(file_no):
        if covers[i][cover_idx] and i!=cover_idx: #if cover and not the same work
            cvrs.append(cover_idx)
    d = dtw_cost[i]
    d = np.argsort(d)
    hits = []
    for c in range(len(cvrs)): #traverse covers
        hits.append(np.where(d==c)[0][0])
    hit_positions.append(min(hits))
dtw_average_hit = np.mean(hit_positions)
print('dtw mean position of first hit:', dtw_average_hit)

#Mean Average Precision
for i in range(file_no):
    #get all distances to selected song, normalize [0,1], convert to similarity metric, not dissimilarity
    d = 1-(dtw_cost[i]/np.linalg.norm(dtw_cost[i])) 
    c = covers[1] #get all cover relationships to selected song
    mAP = 0
    for j in range(file_no):
        mAP += average_precision_score(c, d)
    mAP = mAP/float(file_no)
print('dtw mean average precision:', mAP)


#---Directed Hausdorff distance---#

hausdorff_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        hausdorff_distances[i][j] = (directed_hausdorff(all_flat[i], all_flat[j]))[0]

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
plt.hist(hausdorff_distances_covers, bins=200, alpha=0.5, label='Covers', density=1)
plt.hist(hausdorff_distances_noncovers, bins=200, alpha=0.5, label='Non-covers', density=1)
plt.title("Histogram of Hausdorff distances between cover and non-cover pairs")
plt.legend(loc='upper right')
plt.savefig(fig_dir+'Histogram-hau.png')

#Mean position of first hit
hit_positions = []
for i in range(file_no):
    cvrs = [] #list of cover indeces for that work
    for cover_idx in range(file_no):
        if covers[i][cover_idx] and i!=cover_idx: #if cover and not the same work
            cvrs.append(cover_idx)
    d = hausdorff_distances[i]
    d = np.argsort(d)
    hits = []
    for c in range(len(cvrs)): #traverse covers
        hits.append(np.where(d==c)[0][0])
    hit_positions.append(min(hits))
hau_average_hit = np.mean(hit_positions)
print('hau mean position of first hit:', hau_average_hit)

#Mean Average Precision
for i in range(file_no):
    #get all distances to selected song, normalize [0,1], convert to similarity metric, not dissimilarity
    d = 1-(hausdorff_distances[i]/np.linalg.norm(hausdorff_distances[i])) 
    c = covers[1] #get all cover relationships to selected song
    mAP = 0
    for j in range(file_no):
        mAP += average_precision_score(c, d)
    mAP = mAP/float(file_no)
print('hau mean average precision:', mAP)


#---Minimum distance across all pairs---#

min_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        dists = []
        for n in range(kmax-kmin):
            for m in range(kmax-kmin):
                dists.append(np.linalg.norm(all_sets[i][n]-all_sets[j][m]))
        min_distances[i][j] = min(dists)

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
plt.hist(min_distances_covers, bins=200, alpha=0.5, label='Covers', density=1)
plt.hist(min_distances_noncovers, bins=200, alpha=0.5, label='Non-covers', density=1)
plt.title("Histogram of min pair distances between cover and non-cover pairs")
plt.legend(loc='upper right')
plt.savefig(fig_dir+'Histogram-pair.png')

#Mean position of first hit
hit_positions = []
for i in range(file_no):
    cvrs = [] #list of cover indeces for that work
    for cover_idx in range(file_no):
        if covers[i][cover_idx] and i!=cover_idx: #if cover and not the same work
            cvrs.append(cover_idx)
    d = min_distances[i]
    d = np.argsort(d)
    hits = []
    for c in range(len(cvrs)): #traverse covers
        hits.append(np.where(d==c)[0][0])
    hit_positions.append(min(hits))
pair_average_hit = np.mean(hit_positions)
print('pair mean position of first hit:', pair_average_hit)

#Mean Average Precision
for i in range(file_no):
    #get all distances to selected song, normalize [0,1], convert to similarity metric, not dissimilarity
    d = 1-(min_distances[i]/np.linalg.norm(min_distances[i])) 
    c = covers[1] #get all cover relationships to selected song
    mAP = 0
    for j in range(file_no):
        mAP += average_precision_score(c, d)
    mAP = mAP/float(file_no)
print('pair mean average precision:', mAP)


#---Directed Hausdorff distance shingled tuples---#

shingled2_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        shingled2_distances[i][j] = (directed_hausdorff(all_shingled2[i], all_shingled2[j]))[0]

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
plt.hist(shingled2_distances_covers, bins=200, alpha=0.5, label='Covers', density=1)
plt.hist(shingled2_distances_noncovers, bins=200, alpha=0.5, label='Non-covers', density=1)
plt.title("Histogram of Hausdorff distances between cover and non-cover pairs of shingled tuples")
plt.legend(loc='upper right')
plt.savefig(fig_dir+'Histogram-sh2.png')

#Mean position of first hit
hit_positions = []
for i in range(file_no):
    cvrs = [] #list of cover indeces for that work
    for cover_idx in range(file_no):
        if covers[i][cover_idx] and i!=cover_idx: #if cover and not the same work
            cvrs.append(cover_idx)
    d = shingled2_distances[i]
    d = np.argsort(d)
    hits = []
    for c in range(len(cvrs)): #traverse covers
        hits.append(np.where(d==c)[0][0])
    hit_positions.append(min(hits))
sh2_average_hit = np.mean(hit_positions)
print('sh2 mean position of first hit:', sh2_average_hit)

#Mean Average Precision
for i in range(file_no):
    #get all distances to selected song, normalize [0,1], convert to similarity metric, not dissimilarity
    d = 1-(shingled2_distances[i]/np.linalg.norm(shingled2_distances[i])) 
    c = covers[1] #get all cover relationships to selected song
    mAP = 0
    for j in range(file_no):
        mAP += average_precision_score(c, d)
    mAP = mAP/float(file_no)
print('sh2 mean average precision:', mAP)


#---Directed Hausdorff distance shingled triples---#

shingled3_distances = np.zeros((file_no, file_no))
for i in range(file_no):
    for j in range(file_no):
        shingled3_distances[i][j] = (directed_hausdorff(all_shingled3[i], all_shingled3[j]))[0]

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
plt.hist(shingled3_distances_covers, bins=200, alpha=0.5, label='Covers', density=1)
plt.hist(shingled3_distances_noncovers, bins=200, alpha=0.5, label='Non-covers', density=1)
plt.title("Histogram of Hausdorff distances between cover and non-cover pairs of shingled tuples")
plt.legend(loc='upper right')
plt.savefig(fig_dir+'Histogram-sh3.png')

#Mean position of first hit
hit_positions = []
for i in range(file_no):
    cvrs = [] #list of cover indeces for that work
    for cover_idx in range(file_no):
        if covers[i][cover_idx] and i!=cover_idx: #if cover and not the same work
            cvrs.append(cover_idx)
    d = shingled3_distances[i]
    d = np.argsort(d)
    hits = []
    for c in range(len(cvrs)): #traverse covers
        hits.append(np.where(d==c)[0][0])
    hit_positions.append(min(hits))
sh3_average_hit = np.mean(hit_positions)
print('sh3 mean position of first hit:', sh3_average_hit)

#Mean Average Precision
for i in range(file_no):
    #get all distances to selected song, normalize [0,1], convert to similarity metric, not dissimilarity
    d = 1-(shingled3_distances[i]/np.linalg.norm(shingled3_distances[i])) 
    c = covers[1] #get all cover relationships to selected song
    mAP = 0
    for j in range(file_no):
        mAP += average_precision_score(c, d)
    mAP = mAP/float(file_no)
print('sh3 mean average precision:', mAP)

