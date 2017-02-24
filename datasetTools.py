import numpy as np
import cv2
import os
from random import shuffle
from config import imageSize, datasetPath
import h5py
from imageTools import toData, toImage

# Loads the dataset
def loadDataset():
    datasetName = "coins_64.h5"
    if os.path.isfile(datasetPath+datasetName):
        # Load hdf5 dataset
        h5f = h5py.File(datasetPath+datasetName, 'r')
        X_train = h5f['X_train']
        X_test = h5f['X_test']
        return X_train, X_test
    else:
        #We don't generate the dataset in this example
        print "[!] No dataset found ({})".format(datasetName)
        return None

# Add noise to batch
def addNoise(imgBatch):
    noise = np.zeros(imgBatch.shape[1:], np.float)
    for i in range(imgBatch.shape[0]):
        m, s = (0,0,0), (.12,.12,.12)
        cv2.randn(noise,m,s);
        noise = np.clip(noise,-1.,1.)
        imgBatch[i,:] += noise
    imgBatch = np.clip(imgBatch,-1.,1.)
    return imgBatch

# Displays dataset
def visualizeDataset(X):
    for imgData in X:
        img = toImage(imgData)
        cv2.imshow("LOL",img)
        cv2.waitKey()

# Soft labels
def getTrueLabels(batchSize, soft=False, flipped=False):
    if flipped:
        return getFakeLabels(batchSize, soft=soft)
    else:
        if soft:
            return np.random.uniform(.8, 1., batchSize)
        else:
            return np.ones(batchSize)

# Soft labels
def getFakeLabels(batchSize, soft=False):
    if soft:
        return np.random.uniform(0., .2, (batchSize))
    else:
        return np.zeros(batchSize)
        
if __name__ == "__main__":
    X_train, _ = loadDataset()
    visualizeDataset(X_train)









