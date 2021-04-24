#Importing
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
import glob
import os
import random
import soundfile as sf

all_dirs = []
all_names = []
all_roots = []
max_files = 4000
for root, dirs, files in os.walk('/Users/chris/Google Drive/Classes/Capstone/Datasets/deformations'):
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
    y, sr = librosa.load(filedir, sr=16000, mono=True)
    #trim start
    sf.write(fileroot+'/T03S-'+filename+'.wav', y[3*16000:], sr, subtype='FLOAT')
    sf.write(fileroot+'/T07S-'+filename+'.wav', y[7*16000:], sr, subtype='FLOAT')
    sf.write(fileroot+'/T15S-'+filename+'.wav', y[15*16000:], sr, subtype='FLOAT')
    #trim end
    sf.write(fileroot+'/T03E-'+filename+'.wav', y[:len(y)-(3*16000)], sr, subtype='FLOAT')
    sf.write(fileroot+'/T07E-'+filename+'.wav', y[:len(y)-(7*16000)], sr, subtype='FLOAT')
    sf.write(fileroot+'/T15E-'+filename+'.wav', y[:len(y)-(15*16000)], sr, subtype='FLOAT')
    #add silence to start
    sf.write(fileroot+'/S03S-'+filename+'.wav', np.concatenate((np.zeros(3*16000), y)), sr, subtype='FLOAT')
    sf.write(fileroot+'/S07S-'+filename+'.wav', np.concatenate((np.zeros(7*16000), y)), sr, subtype='FLOAT')
    sf.write(fileroot+'/S15S-'+filename+'.wav', np.concatenate((np.zeros(15*16000), y)), sr, subtype='FLOAT')
    #add silence to end
    sf.write(fileroot+'/S03E-'+filename+'.wav', np.concatenate((y, np.zeros(3*16000))), sr, subtype='FLOAT')
    sf.write(fileroot+'/S07E-'+filename+'.wav', np.concatenate((y, np.zeros(7*16000))), sr, subtype='FLOAT')
    sf.write(fileroot+'/S15E-'+filename+'.wav', np.concatenate((y, np.zeros(15*16000))), sr, subtype='FLOAT')


