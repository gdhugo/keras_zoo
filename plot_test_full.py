#!/usr/bin/env python

""" plot results of keras_zoo/config/test_full.py
"""

from os.path import join
from os import path, mkdir, getcwd, chdir, makedirs
import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Entry point of the script
if __name__ == "__main__":

    parser = ArgumentParser(description='Model testing')
    parser.add_argument('-p', '--path', type=str, help='path to .npy file')
    #parser.add_argument('-d', '--nodisplay', action='store_true', help='Turn off visual display')
    parser.add_argument('-s', '--sample', type=int, help='sample index to plot')
    args = parser.parse_args()

    y_pred = np.load(join(args.path, 'y_pred.npy'))
    plt.imshow(y_pred[args.sample,:,:,0])
    plt.show()
