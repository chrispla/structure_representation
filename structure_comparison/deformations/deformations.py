#Importing
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
import glob
import os
import soundfile as sf
from segment_transformation import segment_cluster
#--supress warnings--#
import warnings
warnings.filterwarnings("ignore")

all_dirs = []
all_names = []
all_roots = []
max_files = 4000
for root, dirs, files in os.walk('/home/chris/Documents/datasets/test2/'):
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

#format: T (Trim) / S (Silence) - Seconds - S (Start) / E (End)
for f in range(file_no):
    filedir = all_dirs[f]
    fileroot = all_roots[f]
    filename = all_names[f]

    #load file
    y, sr = librosa.load(filedir, sr=16000, mono=True)

    #----------------------------#
    #---Simple transformations---#
    #----------------------------#

    #---trim start---#
    sf.write(fileroot+'/T03S-'+filename+'.wav', y[3*16000:], sr, subtype='FLOAT')
    sf.write(fileroot+'/T07S-'+filename+'.wav', y[7*16000:], sr, subtype='FLOAT')
    sf.write(fileroot+'/T15S-'+filename+'.wav', y[15*16000:], sr, subtype='FLOAT')
    #---trim end---#
    sf.write(fileroot+'/T03E-'+filename+'.wav', y[:len(y)-(3*16000)], sr, subtype='FLOAT')
    sf.write(fileroot+'/T07E-'+filename+'.wav', y[:len(y)-(7*16000)], sr, subtype='FLOAT')
    sf.write(fileroot+'/T15E-'+filename+'.wav', y[:len(y)-(15*16000)], sr, subtype='FLOAT')
    #---add silence to start---#
    sf.write(fileroot+'/S03S-'+filename+'.wav', np.concatenate((np.zeros(3*16000), y)), sr, subtype='FLOAT')
    sf.write(fileroot+'/S07S-'+filename+'.wav', np.concatenate((np.zeros(7*16000), y)), sr, subtype='FLOAT')
    sf.write(fileroot+'/S15S-'+filename+'.wav', np.concatenate((np.zeros(15*16000), y)), sr, subtype='FLOAT')
    #---add silence to end---#
    sf.write(fileroot+'/S03E-'+filename+'.wav', np.concatenate((y, np.zeros(3*16000))), sr, subtype='FLOAT')
    sf.write(fileroot+'/S07E-'+filename+'.wav', np.concatenate((y, np.zeros(7*16000))), sr, subtype='FLOAT')
    sf.write(fileroot+'/S15E-'+filename+'.wav', np.concatenate((y, np.zeros(15*16000))), sr, subtype='FLOAT')

    #---------------------------------------#
    #---Structure-dependent tranformations--#
    #---------------------------------------#

    #get frames of laplacian segmentation boundaries
    boundary_frames = segment_cluster(filedir, 128, 3, 4, True)[0]

    #find largest and second largest segment
    large1 = [0, 0] #stant and end frame of largest segment
    large2 = [0, 0] #start and end frame of second largest segment
    for i in range(len(boundary_frames)-1):
        #check for largest
        if ((boundary_frames[i+1]-boundary_frames[i]) >= (large1[1]-large1[0])):
            #move largest to second largest
            large2[0] = large1[0]
            large2[1] = large1[1]
            #set new as largest
            large1[0] = boundary_frames[i]
            large1[1] = boundary_frames[i+1]
        elif (boundary_frames[i+1]-boundary_frames[i] >= (large2[1]-large2[0])):
            #set new as second largest
            large2[0] = boundary_frames[i]
            large2[1] = boundary_frames[i+1]

    #---remove largest structural segment---#
    #concatenate part before start and after end of largest (or second largest) segment
    sf.write(fileroot+'/REM1-'+filename+'.wav', np.concatenate((y[:large1[0]], y[large1[1]:])), sr, subtype='FLOAT')
    sf.write(fileroot+'/REM2-'+filename+'.wav', np.concatenate((y[:large2[0]], y[large2[1]:])), sr, subtype='FLOAT')

    #---duplicate largest structural segment in place---#
    #concatenate start of piece up to end of largest segment with start of largest segment up to end of piece
    sf.write(fileroot+'/DUP1-'+filename+'.wav', np.concatenate((y[:large1[1]], y[large1[0]:])), sr, subtype='FLOAT')
    sf.write(fileroot+'/DUP2-'+filename+'.wav', np.concatenate((y[:large2[1]], y[large2[0]:])), sr, subtype='FLOAT')

    #---swap two largest structural segments---#

    #if largest segment is before second largest
    if large1[0] < large2[0]:
        sf.write(fileroot+'/SWAP-'+filename+'.wav', np.concatenate((y[:large1[0]], #up to start of large1
                                                                    y[large2[0]:large2[1]], #large2
                                                                    y[large1[1]:large2[0]], #between large1 and large2 
                                                                    y[large1[0]:large1[1]], #large 1
                                                                    y[large2[1]:])), #after large2
                                                                    sr, subtype='FLOAT')
    #if largest segment is after second largest
    else:
        sf.write(fileroot+'/SWAP-'+filename+'.wav', np.concatenate((y[:large2[0]], #up to start of large2
                                                                y[large1[0]:large1[1]], #large1
                                                                y[large2[1]:large1[0]], #between large2 and large1
                                                                y[large2[0]:large2[1]], #large 2
                                                                y[large1[1]:])), #after large1
                                                                sr, subtype='FLOAT')

    #progress
    sys.stdout.write("\rComputed transformations of %i/%s pieces." % ((f+1), str(file_no)))
    sys.stdout.flush()
