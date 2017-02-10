import numpy as np
import math
from PIL import Image
import cv2

# Puts generated images in a grid for display
def combine_images(generated_images):
    nbImages = generated_images.shape[0]
    # Hope it's a square
    length = int(math.sqrt(nbImages))
    image = np.zeros([28*length,28*length,3])
    for i in range(length):
        for j in range(length):
            generated_image = generated_images[i+length*j]
            image[i*28:i*28+28,j*28:j*28+28,:] = generated_images[i+length*j]
    return image

# Saves 128 generates images on disk
def saveGeneratedImages(generated_images, name):
    img = combine_images(generated_images)
    img = img*255.
    cv2.imwrite("Images/"+name+".png",img.astype(np.uint8))