""" 
Useful links
http://torch.ch/blog/2015/11/13/gan.html
https://github.com/Newmu/dcgan_code
https://github.com/soumith/ganhacks
"""

import argparse
from model import getGenerator, getDiscriminator, getGeneratorContainingDiscriminator
from datasetTools import loadDataset, getTrueLabels, getFakeLabels
import numpy as np

from keras.optimizers import SGD, Adam
from imageTools import saveGeneratedImages, toData, toImage
from datasetTools import loadDataset, addNoise

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

# Trains GAN
def train(batchSize):
    # Load data
    X_train, X_test = loadDataset()
    
    # Create models
    discriminator = getDiscriminator()
    generator = getGenerator()
    discriminator_on_generator = getGeneratorContainingDiscriminator(generator, discriminator)
    
    # Create optimizers for generator and discriminator
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = Adam(lr=0.0005)
    
    # Compile models with optimizers
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    
    # Prepare 100D noise matrix for each batch
    # We keep the same noise to follow the generation
    displayNoise = np.random.normal(0, 1, (9, 100))
        
    # Torch.ch heuristic for training
    lossMargin = 0.3
    discriminator.trainable = True
    generator.trainable = True

    epochNb = 300
    # Compute number of batches
    batchNb = int(X_train.shape[0]/batchSize)

    print("Starting training for {} epochs with {} batches of size {}".format(epochNb, batchNb, batchSize))

    # For each epoch
    for epoch in range(epochNb):        
        # For each batch
        for batchIndex in range(batchNb):

            # Save images to disk every batch
            generated_images = generator.predict(displayNoise, verbose=0)
            saveGeneratedImages(generated_images, "{}_{}".format(epoch, batchIndex))

            ##### DISCRIMINATOR TRAINING #####
            # Generate new noise for discriminator training
            noise = np.random.normal(0, 1, (batchSize, 100))

            # Get real image batch from data
            real_images = X_train[batchIndex*batchSize:(batchIndex+1)*batchSize]
            # Add noise to make it harder for discriminator
            X = addNoise(real_images)
            # Soft labels
            y = getTrueLabels(batchSize) 
            # Train on real images, or just compute loss if not trainable
            d_loss_real = discriminator.train_on_batch(X, y)

            # Generate a batch of fake images
            generated_images = generator.predict(noise, verbose=0)
            # Add noise to make it harder for discriminator
            X = addNoise(generated_images)
            # Soft labels
            y = getFakeLabels(batchSize)
            # Train on fake images, or just compute loss if not trainable
            d_loss_fake = discriminator.train_on_batch(X, y)

            ##### GENERATOR TRAINING #####
            # Generator is trained twice because the discriminator is
            for i in range(2):
                # Generate new noise for generator training
                noise = np.random.uniform(-1, 1, (batchSize, 100))
                # Trick: max(log(D)) instead of (min(log(1-D)))
                y = getTrueLabels(batchSize, flipped=True)
                shouldDiscriminatorBeTrained = discriminator.trainable
                # Always put as not trainable
                discriminator.trainable = False
                # Train generator on real batch, or just compute loss if not trainable
                g_loss = discriminator_on_generator.train_on_batch(noise, y)
                # Restore true value of trainable
                discriminator.trainable = shouldDiscriminatorBeTrained
            
            print("Epoch {}/{} - Batch {}/{} - (G: {:.3f}) - (D_true: {:.3f}, D_fake: {:.3f})".format(epoch+1, epochNb, batchIndex+1, batchNb, g_loss, d_loss_real, d_loss_fake))
            
            # Assume both should be trained good
            discriminator.trainable = True
            generator.trainable = True

            # Heuristic from Torch.ch
            # Discriminator too powerful
            if d_loss_fake < lossMargin or d_loss_real < lossMargin:
                print "Stopping D for next update"
                discriminator.trainable = False
            
            # Discriminator too weak
            if d_loss_fake > (1.0-lossMargin) or d_loss_real > (1.0-lossMargin):
                print "Stopping G for next update"
                generator.trainable = False
            
            # Both are too good, train both
            if not discriminator.trainable and not generator.trainable:
                print "Starting G and D"
                discriminator.trainable = True
                generator.trainable = True

            if batchIndex%40 == 0:
                #Save model every N batches
                print("Saving models...")
                generator.save_weights('Models/generator', True)
                discriminator.save_weights('Models/discriminator', True)


"""
# Generates new samples
def generate(batchSize, nice=False):
    # Create, compile and load generator
    generator = generator()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('Models/generator')

    # Top 5% images according to discriminator
    if nice:
        # Create, compile and load discriminator
        discriminator = discriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('Models/discriminator')
        
        # Create noise for each batch
        noise = np.random.uniform(-1, 1, [batchSize*20, 100])

        # Generate image batches based on noise
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, batchSize*20)
        index.resize((batchSize*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)

        # Placeholder for images
        nice_images = np.zeros((batchSize, 1) + (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(batchSize)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        # Create noise for each batch
        noise = np.random.uniform(-1, 1, (batchSize, 100))

        # Generate image batches based on noise
        generated_images = generator.predict(noise, verbose=1)
        
        # Write generated images on disk
        saveGeneratedImages(generated_images,"generated_Images")
"""


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batchSize=args.batchSize)
    elif args.mode == "generate":
        generate(batchSize=args.batchSize, nice=args.nice)

