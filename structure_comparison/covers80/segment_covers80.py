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