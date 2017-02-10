import numpy as np
import cv2
import os
from random import shuffle

# Loads the dataset
def load_dataset():
    #Load, normalize and reshape
    files = os.listdir("Data/")
    files = [file for file in files if file.endswith(".jpg")]
    
    includeDisabled = False
    if not includeDisabled:
        files = [file for file in files if not file.startswith("DIS")]

    shuffle(files)
    X_train = []
    for file in files:
        img = cv2.imread("Data/"+file, 1)
        img = cv2.resize(img,(28,28))
        img = img.astype(float)/255.
        X_train.append(img)

    X_train = np.array(X_train)
    return X_train

def addNoise(img):
    noise = np.zeros(img.shape[1:], np.float)
    for i in range(img.shape[0]):
        m, s = (0,0,0), (.15,.15,.15)
        cv2.randn(noise,m,s);
        noise = np.clip(noise,0.,1.)
        img[i,:] += noise

    img = np.clip(img,0.,1.)
    return img

# Loads the dataset
def explore_data():
    #Load, normalize and reshape
    files = os.listdir("Data/")
    files = [file for file in files if file.endswith(".jpg")]
    
    includeDisabled = False
    if not includeDisabled:
        files = [file for file in files if not file.startswith("DIS")]

    file = files[0]
    img = cv2.imread("Data/"+file, 1)
    img = cv2.resize(img,(28,28))
    img = img.astype(float)/255.

    while True:
        noisyImg = addNoise(img);

        #img = img + noise
        cv2.imshow("LOL",noisyImg)
        cv2.waitKey()


if __name__ == "__main__":
    #load_dataset()
    explore_data()




