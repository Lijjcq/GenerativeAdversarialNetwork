import numpy as np
import math
from PIL import Image
import cv2
from config import imageSize, generatedImagesPath

# From [0;255] to [-1;1]
def toData(img):
    return img.astype(float) * 2/255. - 1.

# From [-1;1] to [0;255]
def toImage(img):
    img = (img+1.)*255./2.
    return img.astype(np.uint8)

# Puts generated images in a grid for display
def combine_images(generated_images):
    nbImages = generated_images.shape[0]
    # Hope it's a square
    length = int(math.sqrt(nbImages))
    image = np.zeros([imageSize*length,imageSize*length,3])
    for i in range(length):
        for j in range(length):
            generated_image = generated_images[i+length*j]
            image[i*imageSize:i*imageSize+imageSize,j*imageSize:j*imageSize+imageSize,:] = generated_images[i+length*j]
    return image

# Saves imageSize generated images on disk
def saveGeneratedImages(generated_images, name):
    img = combine_images(generated_images)
    img = toImage(img)
    cv2.imwrite(generatedImagesPath+name+".png",img)