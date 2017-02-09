import numpy as np
import math
from PIL import Image

# Puts generated images in a grid for display
def combine_images(generated_images):
    nbImages = generated_images.shape[0]
    # Hope it's a square
    length = int(math.sqrt(nbImages))
    image = np.zeros([28*length,28*length,4])
    for i in range(length):
        for j in range(length):
            generated_image = generated_images[i+length*j]
            image[i:i+28,j:j+28,:] = generated_images[i+length*j]
    return image

# Saves 128 generates images on disk
def saveGeneratedImages(generated_images, name):
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save("Images/"+name+".png")