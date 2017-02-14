import numpy as np
import cv2
import os
from random import shuffle
from config import imageSize, datasetPath
import h5py

# Loads the dataset
def load_dataset():

    if os.path.isfile(datasetPath+"data.h5"):
        # Load hdf5 dataset
        h5f = h5py.File(datasetPath+"data.h5", 'r')
        X_train = h5f['X']
        return X_train

    else:
        #Load, normalize and reshape
        files = os.listdir("Data/")
        files = [file for file in files if file.endswith(".png")]
      
        shuffle(files)
        X_train = []
        for file in files:
            img = cv2.imread("Data/"+file, -1)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img = cv2.resize(img,(imageSize,imageSize))
            img = img.astype(float)/255.
            X_train.append(img)

        X_train = np.array(X_train)

        #Save at hdf5 format
        print "Saving dataset in hdf5 format... - this may take a while"
        h5f = h5py.File(datasetPath+"data.h5", 'w')
        h5f.create_dataset('X', data=X_train)
        h5f.close()

        return X_train

# Add noise to batch
def addNoise(img):
    noise = np.zeros(img.shape[1:], np.float)
    for i in range(img.shape[0]):
        m, s = (0,0,0), (.12,.12,.12)
        cv2.randn(noise,m,s);
        noise = np.clip(noise,0.,1.)
        img[i,:] += noise

    img = np.clip(img,0.,1.)
    return img

# Loads the dataset
def explore_data():
    #Load, normalize and reshape
    files = os.listdir("Data/")
    files = [file for file in files if file.endswith(".png")]
    
    file = files[0]
    img = cv2.imread("Data/"+file, -1)
    img = cv2.resize(img,(imageSize,imageSize))
    img = img.astype(float)/255.

    while True:
        noisyImgs = addNoise(np.array([img,img,img]))
        noisyImg = noisyImgs[0]

        #img = img + noise
        cv2.imshow("LOL",noisyImg)
        cv2.waitKey()


if __name__ == "__main__":
    pass
    #extractCIFAR100(display=False)
    load_dataset()
    #explore_data()




