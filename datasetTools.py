import numpy as np
import cv2
import os
from random import shuffle

# Loads the dataset
def load_dataset():
    #Load, normalize and reshape
    files = os.listdir("Data/")
    files = [file for file in files if file.endswith(".png")]
    
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


# Loads the CIFAR-10 dataset
def extractCIFAR10(display=False):
    #Load, normalize and reshape
    files = os.listdir("CIFAR-10/")
    files = [file for file in files if file.startswith("data_batch_")]

    for file in files:
        data = unpickle("CIFAR-10/"+file)
        images, labels = data['data'], data['labels']

        #Keep boats
        for i in range(len(images)):
            label = labels[i]

            if label == 8:
                img = images[i]
                r, g, b = img[:1024],img[1024:2048],img[2048:]
                r = r.reshape([32,32])
                g = g.reshape([32,32])
                b = b.reshape([32,32])
                img = cv2.merge((b,g,r))
                cv2.imwrite("Data/{}_{}.png".format(file,i),img)

                if display:
                    cv2.imshow("LOL",img)
                    cv2.waitKey()


# Loads the CIFAR-10 dataset
def extractCIFAR100(display=False):

    fineLabels =  unpickle("CIFAR-100/meta")['fine_label_names']
    print fineLabels

    selectedClass = fineLabels.index('aquarium_fish')

    #Load, normalize and reshape
    files = os.listdir("CIFAR-100/")
    files = [file for file in files if file.startswith("data_batch_")]

    for file in files:
        data = unpickle("CIFAR-100/"+file)
        images, labels = data['data'], data['fine_labels']
        #Keep boats
        for i in range(len(images)):
            label = labels[i]

            if label == selectedClass:
                img = images[i]
                r, g, b = img[:1024],img[1024:2048],img[2048:]
                r = r.reshape([32,32])
                g = g.reshape([32,32])
                b = b.reshape([32,32])
                img = cv2.merge((b,g,r))
                
                cv2.imwrite("Data/{}_{}.png".format(file,i),img)

                if display:
                    cv2.imshow("LOL",img)
                    cv2.waitKey()

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

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
    extractCIFAR100(display=False)
    #load_dataset()
    #explore_data()




