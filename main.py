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

    epochNb = 10
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
            # Get real image batch from data
            real_images = X_train[batchIndex*batchSize:(batchIndex+1)*batchSize]
            # Add noise to make it harder for discriminator
            X = addNoise(real_images)
            y = getTrueLabels(batchSize) 
            # Train on real images, or just compute loss if not trainable
            d_loss_real = discriminator.train_on_batch(X, y)

            # Generate a batch of fake images
            noise = np.random.normal(0, 1, (batchSize, 100))
            generated_images = generator.predict(noise, verbose=0)
            # Add noise to make it harder for discriminator
            X = addNoise(generated_images)
            y = getFakeLabels(batchSize)
            # Train on fake images, or just compute loss if not trainable
            d_loss_fake = discriminator.train_on_batch(X, y)

            ##### GENERATOR TRAINING #####
            # Generator is trained twice because the discriminator is too
            for i in range(2):
                # Flipped labels trick: max(log(D)) instead of min(log(1-D))
                y = getTrueLabels(batchSize, flipped=False)
                shouldDiscriminatorBeTrained = discriminator.trainable
                # Always put as not trainable first
                discriminator.trainable = False
                # Train generator, or just compute loss if not trainable
                noise = np.random.uniform(-1, 1, (batchSize, 100))
                g_loss = discriminator_on_generator.train_on_batch(noise, y)
                # Restore true value of trainable
                discriminator.trainable = shouldDiscriminatorBeTrained
            
            print("Epoch {}/{} - Batch {}/{} - (G: {:.3f}) - (D_true: {:.3f}, D_fake: {:.3f})".format(epoch+1, epochNb, batchIndex+1, batchNb, g_loss, d_loss_real, d_loss_fake))
            

            # Heuristic from Torch.ch
            # Assume both should be trained
            discriminator.trainable = True
            generator.trainable = True

            # Discriminator too powerful
            if d_loss_fake < lossMargin or d_loss_real < lossMargin:
                print "Stopping D for next update"
                discriminator.trainable = False
            
            # Discriminator too weak
            if d_loss_fake > (1.0-lossMargin) or d_loss_real > (1.0-lossMargin):
                print "Stopping G for next update"
                generator.trainable = False
            
            # Both are good, train both
            if not discriminator.trainable and not generator.trainable:
                print "Starting G and D"
                discriminator.trainable = True
                generator.trainable = True
            
            #Save model every N batches
            if batchIndex%40 == 0:
                print "Saving models..."
                generator.save_weights('Models/generator', True)
                discriminator.save_weights('Models/discriminator', True)

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batchSize=args.batchSize)
    elif args.mode == "generate":
        test(batchSize=args.batchSize)

