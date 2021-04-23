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

dill.load_session('../../../dills/covers80_all.db')

#Plotting scores
fig, ax = plt.subplots()
xticks = list(scores.keys())
all_scores = list(scores.values())

ax.barh(xticks, all_scores)
ax.set(xlabel='Score', ylabel='Construction')
plt.savefig('bar_plot.png')
