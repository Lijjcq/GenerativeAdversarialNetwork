import numpy as np
import cv2
import os
from random import shuffle
from config import imageSize, datasetPath
import h5py

# Loads the dataset
def loadDataset():
    if os.path.isfile(datasetPath+"coins.h5"):
        # Load hdf5 dataset
        h5f = h5py.File(datasetPath+"coins.h5", 'r')
        X_train = h5f['X']
        return X_train[:8000], X_train[-1000:]
    else:
        #We don't generate the dataset in this example
        print "[!] No dataset found (coins.h5)"
        return None

# Add noise to batch
def addNoise(imgBatch):
    noise = np.zeros(imgBatch.shape[1:], np.float)
    for i in range(imgBatch.shape[0]):
        m, s = (0,0,0), (.12,.12,.12)
        cv2.randn(noise,m,s);
        noise = np.clip(noise,0.,1.)
        imgBatch[i,:] += noise
    imgBatch = np.clip(imgBatch,0.,1.)
    return imgBatch