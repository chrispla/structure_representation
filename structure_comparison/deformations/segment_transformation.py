import librosa
import scipy
import numpy as np
import cv2
import sklearn

def segment(filedir, rs_size, kmin, kmax, filter):
    """compute laplacian approximations

        rs_size: side length to which combined matrix is going to be resampled to
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

    #beat synchronization would remove the silence in deformations that have added silence at the start
    #therefore we need to downsample using a different method
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

def segment_cluster(filedir, rs_size, kmin, kmax, filter):
    """compute laplacian approximations and perform spectral clustering

    rs_size: side length to which combined matrix is going to be resampled to
    [kmin, kmax]: min and maximum approximation ranks

    returns list of boundary frames at every specified level of approximation"""

    #load audio
    y, sr = librosa.load(filedir, sr=16000, mono=True)
    y_len = len(y)

    #compute cqt
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, 
                                        hop_length=512,
                                        bins_per_octave=12*3,
                                        n_bins=7*12*3)),
                                        ref=np.max)

    #beat synchronization would remove the silence in deformations that have added silence at the start
    #therefore we need to downsample using a different method
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

    #clustering
    seg_ids_set = []
    for k in range(kmin, kmax):

        Xs = evecs[:, :k] / Cnorm[:, k-1:k]
        
        #debug
        if np.isnan(np.sum(Xs)):
            print('woops')

        KM = sklearn.cluster.KMeans(n_clusters=k, n_init=50, max_iter=500)
        seg_ids = KM.fit_predict(Xs)
        seg_ids_set.append(seg_ids)
    seg_ids_set = np.asarray(seg_ids_set)
    print(seg_ids_set)

    #compute boundary frames for each level
    boundary_frames = []
    for k in range(kmax-kmin):
        #add starting frame
        boundary_frames.append([0])
        print(seg_ids_set.shape)
        for i in range(seg_ids_set.shape[1]-1):
            if seg_ids_set[k][i] != seg_ids_set[k][i+1]:
                #these are representative of all the downsampling performed, so we need to scale back to original
                frame = int((i+0.5)*y_len/seg_ids_set.shape[1])
                boundary_frames[k].append(frame) 
        #add ending frame
        boundary_frames[k].append(y_len)

    #return
    return(boundary_frames)