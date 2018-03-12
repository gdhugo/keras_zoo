#!/usr/bin/env python

""" test script for keras_zoo. Writes data to
    shared_path, including images/masks and config.
    Use the train.py script to run the test, pointing
    the script at the shared_path created here and the
    config in keras_zoo/config/test_full.py
"""

from os.path import join
from os import path, mkdir, getcwd, chdir, makedirs
import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
#matplotlib.use('Agg')  # Faster plot

# Import tools
#from models.fcn8 import build_fcn8
#from metrics.metrics import cce_flatt, IoU
#from keras.optimizers import (RMSprop, Adam, SGD)

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

def write_data(data, write_path, normalize=False):
    """ write out data as PNG
    """
    # check n_channels
    if data.shape[3] is not 1 and data.shape[3] is not 3:
        print("Error: data must be either single or 3 channel. Cannot write to PNG")
        exit()

    import scipy.misc
    for sample in range( data.shape[0] ):
        file_name = join(write_path, 'image' + str(sample) + '.png')
        if normalize:
            scipy.misc.toimage(data[sample,:,:,0]).save(file_name)
        else:
            # find min/max
            scipy.misc.toimage(data[sample,:,:,0], cmin=0.0, cmax=1.0).save(file_name)

def write_config(x_size, y_size, write_path):
    """ write out config file for dataset
    """
    # class_mode = 'segmentation'
    # data_format = 'png'
    # color_mode = 'grayscale'
    # img_shape = [32,32]
    # classes = {"back" : "0", "fore" : "1" }
    # n_classes = 2
    # n_channels = 1
    # void_class = [-1]
    # rgb_mean = 0
    # rgb_std = 1
    file_name = join(write_path, 'config.py')
    with open(file_name, 'w') as f:
        f.write("class_mode = \'segmentation\'\n")
        f.write("data_format = \'folders\'\n")
        f.write("color_mode = \'grayscale\'\n")
        f.write("color_map = [(0.86, 0.3712, 0.33999999999999997), (0.33999999999999997, 0.86, 0.3712), (0.3712, 0.33999999999999997, 0.86)]\n")
        f.write("img_shape = [{0},{1}]\n".format(x_size, y_size))
        f.write("classes = {\'back\' : \'0\', \'fore\' : \'1\'}\n")
        f.write("n_classes = 2\n")
        f.write("n_channels = 1\n")
        f.write("void_class = [-1]\n")
        f.write("rgb_mean = 0\n")
        f.write("rgb_std = 1\n")
        f.write("n_images_train = 125\n")
        f.write("n_images_valid = 24\n")
        f.write("n_images_test = 49\n")

# Entry point of the script
if __name__ == "__main__":

    parser = ArgumentParser(description='Model testing')
    #parser.add_argument('-m', '--nomodel', action='store_true', help='Do not perform modeling?')
    parser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess by pixelwise mean / std centering')
    #parser.add_argument('-d', '--nodisplay', action='store_true', help='Turn off visual display')
    parser.add_argument('-s', '--shared_path', type=str, default='/home/ghugo/data', help='Shared path')
    args = parser.parse_args()

    x_size = 32
    y_size = 32
    n_channels = 1
    n_samples = 200
    (x_train, y_train, x_test, y_test, x_valid, y_valid) = make_data(x_size, y_size, n_channels, n_samples)
    print(x_train.shape)
    print(y_train.shape)

    if(args.preprocess):
        print( np.min(x_train) )
        print( np.max(x_train) )
        print( np.min(x_test) )
        print( np.max(x_test) )
        print( np.min(x_valid) )
        print( np.max(x_valid) )
        x_train_mean = np.mean(x_train, axis=0)
        x_train_std = np.std(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        x_valid -= x_train_mean
        x_train /= (x_train_std + 1e-7)
        x_test /= (x_train_std + 1e-7)
        x_valid /= (x_train_std + 1e-7)
        print( np.min(x_train) )
        print( np.max(x_train) )
        print( np.min(x_test) )
        print( np.max(x_test) )
        print( np.min(x_valid) )
        print( np.max(x_valid) )
        barf
    # output test data

    # make directories
    data_dir = args.shared_path
    print('output directory: ' + data_dir)
    if not path.exists(data_dir):
        try:
            makedirs(data_dir)
        except:
            print("Error:", sys.exc_info()[0])
            exit()

    # subdirs
    subdirs = ['test', 'train', 'valid']
    subsubdirs = ['images','masks']

    for subdir in subdirs:
        for subsubdir in subsubdirs:
            new_path = join(data_dir, subdir, subsubdir)
            if not path.exists(new_path):
                try:
                    makedirs(new_path)
                except:
                    print("Error:", sys.exc_info()[0])
                    exit()

    # write out data
    write_data(x_train, join(data_dir, 'train', 'images'))
    write_data(y_train, join(data_dir, 'train', 'masks'))
    write_data(x_test, join(data_dir, 'test', 'images'))
    write_data(y_test, join(data_dir, 'test', 'masks'))
    write_data(x_valid, join(data_dir, 'valid', 'images'))
    write_data(y_valid, join(data_dir, 'valid', 'masks'))

    # write out config
    write_config(x_size, y_size, data_dir)