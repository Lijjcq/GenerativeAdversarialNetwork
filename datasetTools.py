import numpy as np
import cv2
import os
from keras.datasets import mnist

# Loads the dataset
def load_dataset():
    #Load, normalize and reshape
    files = os.listdir("Data/")
    files = [file for file in files if file.endswith(".png")]
    
    X_train = []
    for file in files:
        img = cv2.imread("Data/"+file,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img = cv2.resize(img,(28,28))
        #print img.shape
        X_train.append(img)

    X_train = np.array(X_train)
    return X_train

# def load_dataset():

#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     X_train = (X_train.astype(np.float32) - 127.5)/127.5
#     X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
#     return X_train, y_train


if __name__ == "__main__":
    load_dataset()