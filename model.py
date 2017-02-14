""" 
Useful links
http://torch.ch/blog/2015/11/13/gan.html
https://github.com/Newmu/dcgan_code
https://github.com/soumith/ganhacks
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
import tensorflow as tf
tf.python.control_flow_ops = tf

import numpy as np
from PIL import Image
from imageTools import saveGeneratedImages
from datasetTools import load_dataset, addNoise
from config import imageSize

import cv2
import os

# Builds the generator (Noise -> Image)
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('elu'))
    model.add(Dropout(0.4))

    # Prepare for 128/8=8 image with 128 channels
    model.add(Dense(imageSize/16*imageSize/16*128))
    model.add(Activation('elu'))
    model.add(Dropout(0.4))

    # Reshape flat to image
    model.add(Reshape([imageSize/16,imageSize/16, 128]))

    # Upsample + conv to imageSize/8
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('elu'))

    # Upsample + conv to imageSize/4
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(Activation('elu'))

    # Upsample + conv to imageSize/2
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.4))

    # Upsample + conv
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

# Builds the discriminator (Image -> True/Fake)
def discriminator_model():
    model = Sequential()
    # Conv + pool
    model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=(imageSize, imageSize, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    # Conv + pool
    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv + pool
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv + pool
    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    # Fully connected
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('elu'))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

# Builds whole network with only generator trainable
def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

# Trains GAN
def train(BATCH_SIZE):
    # Load data
    X_train = load_dataset()
    
    # Create models
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    
    # Create optimizers for generator and discriminator
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = Adam(lr=0.005)
    
    # Compile models with optimizers
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    
    # Prepare 100D noise matrix for each batch
    # We keep the same noise to follow the generation
    displayNoise = np.zeros((9, 100))
    for i in range(9):
        displayNoise[i, :] = np.random.uniform(-1, 1, 100)
            
    # Prepare 100D noise matrix for each batch
    noise = np.zeros((BATCH_SIZE, 100))

    trainGenerator = True
    trainDiscrimnator = True
    lossMargin = 0.4

    # For each epoch
    for epoch in range(300):
        # Compute number of batches
        batchNb = int(X_train.shape[0]/BATCH_SIZE)
        print("Epoch is", epoch)
        print("Number of batches", batchNb)
        
        # For each batch
        for batchIndex in range(batchNb):

            # Save images to disk every batch
            generated_images = generator.predict(displayNoise, verbose=0)
            saveGeneratedImages(generated_images, "{}_{}".format(epoch, batchIndex))

            ##### DISCRIMINATOR TRAINING #####
            # Generate new noise for discriminator training
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            
            # Get real image batch from data
            real_images = X_train[batchIndex*BATCH_SIZE:(batchIndex+1)*BATCH_SIZE]
            # Add noise to make it harder for discriminator
            X = addNoise(real_images)
            y = np.ones(BATCH_SIZE)
            # Train on real images
            d_loss_real = discriminator.train_on_batch(X, y)

            # Generate a batch of fake images
            generated_images = generator.predict(noise, verbose=0)
            # Add noise to make it harder for discriminator
            X = addNoise(generated_images)
            y = np.zeros(BATCH_SIZE)
            # Train on fake iimages
            d_loss_fake = discriminator.train_on_batch(X, y)

            ##### GENERATOR TRAINING #####
            # Generator is trained twice because the discriminator is
            for i in range(2):
                # Generate new noise for generator training
                for i in range(BATCH_SIZE):
                    noise[i, :] = np.random.uniform(-1, 1, 100)

                # Train generator on real batch
                discriminator.trainable = False
                g_loss = discriminator_on_generator.train_on_batch(noise, np.ones(BATCH_SIZE))
                discriminator.trainable = True

            
            print("Epoch %d, batch %d, g_loss : %f, d_loss_true: %f, d_loss_fake: %f" % (epoch, batchIndex, g_loss, d_loss_real, d_loss_fake))
            
            # Heuristic from Torch.ch
            # Discriminator too powerful
            if d_loss_fake < margin or d_loss_real < margin:
                trainDiscrimnator = False
            
            # Discriminator too weak
            if d_loss_fake > (1.0-margin) or d_loss_real > (1.0-margin) then
                trainGenerator = False
            
            # Both are too good, train both
            if not trainDiscrimnator and not trainGenerator:
                trainDiscrimnator = True
                trainGenerator = True

        #Save model every epoch
        print("Saving models...")
        generator.save_weights('Models/generator', True)
        discriminator.save_weights('Models/discriminator', True)

# Generates new samples
def generate(BATCH_SIZE, nice=False):
    # Create, compile and load generator
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('Models/generator')

    # Top 5% images according to discriminator
    if nice:
        # Create, compile and load discriminator
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('Models/discriminator')
        
        # Create noise for each batch
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)

        # Generate image batches based on noise
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)

        # Placeholder for images
        nice_images = np.zeros((BATCH_SIZE, 1) + (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        # Create noise for each batch
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        
        # Generate image batches based on noise
        generated_images = generator.predict(noise, verbose=1)
        
        # Write generated images on disk
        saveGeneratedImages(generated_images,"generated_Images")








        