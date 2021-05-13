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
with open('./da-tacos_metadata/da-tacos_benchmark_subset_metadata.json') as f:
    benchmark_metadata = json.load(f)

#---Segmentation parameters---#
rs_size = 128
approx = [4,8]

#---Counters---#
count = 0
W_count=0
P_count = 0

#---Loading limits---#
min_covers = 5 #load works for which there are at least min_covers performances
max_covers = 5000 #stop loading performances if over max_covers per work

#---Storage---#
all_sets = []
all_shapeDNAs = []
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
                SSM = dd.io.load("./da-tacosSSMs/StructureLaplacian_datacos_crema_" + P + ".h5")['WFused']
            except:
                print("Couldn't load " + P + ".")
                continue

            N = dd.io.load("./da-tacosSSMs/StructureLaplacian_datacos_crema_" + P + ".h5")['N']

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

            #Shape DNA
            shapeDNA = evals[:30]
            all_shapeDNAs.append(shapeDNA)

            #Hierarchical structure
            evecs = median_filter(evecs, size=(9, 1))
            Cnorm = np.cumsum(evecs**2, axis=1)**0.5
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
        sys.stdout.write("\rLoading %i/100 works." % W_count)
        sys.stdout.flush()
        if (W_count >= 100):
            break
            
all_sets = np.asarray(all_sets)
all_shapeDNAs = np.asarray(all_shapeDNAs)

print("\nLoaded Da-TACOS SMMs.")
print("Data shape:", all_sets.shape)