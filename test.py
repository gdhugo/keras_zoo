#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
#matplotlib.use('Agg')  # Faster plot

# Import tools
from models.fcn8 import build_fcn8
from metrics.metrics import cce_flatt, IoU
from keras.optimizers import (RMSprop, Nadam, SGD)
from keras.callbacks import EarlyStopping

def make_data(x_size, y_size, n_channels, n_samples):
    """ makes testing and training data from scratch
    returns x_train, y_train, x_test, y_test, x_valid, y_valid
    """
    in_shape = (x_size, y_size, n_channels)
    x = np.zeros((n_samples, x_size, y_size, n_channels))
    y = np.zeros((n_samples, x_size, y_size, n_channels))
    n_classes = 2
    void_class = [-1]

    noise_data = (np.random.rand(n_samples, x_size, y_size, n_channels)-0.5) * 0.1 #[-0.05,0.05]
    x_box_size = 10
    y_box_size = 10
    x_box_starts = np.floor( np.random.rand(n_samples) * (x_size - x_box_size)).astype('int')
    y_box_starts = np.floor( np.random.rand(n_samples) * (y_size - y_box_size)).astype('int')
    for sample in range(n_samples):

        box_slice = np.zeros(in_shape)
        box_slice[x_box_starts[sample]:x_box_starts[sample]+x_box_size,y_box_starts[sample]:y_box_starts[sample]+y_box_size,:] = 1.

        y[sample,:,:,:] = box_slice
        x[sample,:,:,:] = box_slice * 2. + noise_data[sample,:,:,:] + 0.05
        x[sample,:,:,:] = x[sample,:,:,:] / 2.10 # normalize [0,1]


    x_train = x[:125,:,:,:]
    y_train = y[:125,:,:,:]
    x_valid = x[126:150,:,:,:]
    y_valid = y[126:150,:,:,:]
    x_test = x[151:,:,:,:]
    y_test = y[151:,:,:,:]
    return x_train, y_train, x_test, y_test, x_valid, y_valid

# Entry point of the script
if __name__ == "__main__":

    parser = ArgumentParser(description='Model testing')
    parser.add_argument('-m', '--nomodel', action='store_true', help='Do not perform modeling?')
    parser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess by pixelwise mean / std centering')
    parser.add_argument('-d', '--nodisplay', action='store_true', help='Turn off visual display')
    args = parser.parse_args()

    x_size = 32
    y_size = 32
    n_channels = 1
    n_samples = 200
    void_class = [-1]
    n_classes = 2
    in_shape = (x_size, y_size, n_channels)
    (x_train, y_train, x_test, y_test, x_valid, y_valid) = make_data(x_size, y_size, n_channels, n_samples)
    print(x_train.shape)
    print(y_train.shape)

    if(args.preprocess):
        x_train_mean = np.mean(x_train, axis=0)
        x_train_std = np.std(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        x_train /= (x_train_std + 1e-7)
        x_test /= (x_train_std + 1e-7)

    # plot data
    if(not args.nodisplay):
        for idx in range(25):
            plt.subplot(5,10,2*idx+1)
            plt.imshow(x_train[idx,:,:,0])
            plt.subplot(5,10,2*idx+2)
            plt.imshow(y_train[idx,:,:,0])
        plt.show()

    if(not args.nomodel):
        loss = cce_flatt(void_class, None)
        metrics = [IoU(n_classes, void_class)]
        #opt = RMSprop(lr=0.001, clipnorm=10)
        opt = Nadam(lr=0.002)

        model = build_fcn8(in_shape, n_classes, 0.)
        model.compile(loss=loss, metrics=metrics, optimizer=opt)

        cb = [EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience=2)]
        model.fit(x_train, y_train, epochs=1000, batch_size=16, callbacks=cb, validation_data=(x_valid,y_valid))

        score = model.evaluate(x_test, y_test) #, batch_size=128)
        y_pred = model.predict(x_test)

        print(score)

        for sample in range(y_test.shape[0]):
            print('sample: ' + str(sample))
            print('actual:')
            print(y_test[sample,:,:,0])
            print('predicted:')
            print(y_pred[sample,:,:,0])

        np.save('y_test.numpy',y_test)
        np.save('y_pred.numpy',y_pred)

        if(not args.nodisplay):
            plt.subplot(1,2,1)
            plt.imshow(y_test[-1,:,:,0])
            plt.subplot(1,2,2)
            plt.imshow(y_pred[-1,:,:,0])
            plt.show()
    else:
        y_test = np.load('../y_test.numpy.npy')
        y_pred = np.load('../y_pred.numpy.npy')
        plt.subplot(1,3,1)
        plt.imshow(y_test[0,:,:,0])
        plt.subplot(1,3,2)
        plt.imshow(y_pred[0,:,:,0])
        plt.subplot(1,3,3)
        plt.imshow(y_pred[0,:,:,1])
        plt.show()
