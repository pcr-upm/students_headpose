#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database
from images_framework.src.composite import Composite
from images_framework.alignment.students_headpose.src.students_headpose import StudentsHeadpose
from images_framework.alignment.students_headpose.src.load_db import load_annotations
import warnings
warnings.filterwarnings("ignore")


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns-train', '-t', dest='anns_train', required=True,
                        help='Ground truth annotations file for training.')
    parser.add_argument('--anns-valid', '-v', dest='anns_valid', required=True,
                        help='Ground truth annotations file for validation.')
    args, unknown = parser.parse_known_args()
    anns_train = args.anns_train
    anns_valid = args.anns_valid
    return unknown, anns_train, anns_valid

def main():
    """
    Students headpose train script.
    """
    unknown, anns_train, anns_valid = parse_options()
    # Load computer vision components
    composite = Composite()
    sa = StudentsHeadpose('images_framework/alignment/students_headpose/')
    composite.add(sa)

    composite.parse_options(unknown)
    anns_train = load_annotations(anns_train)
    anns_valid = load_annotations(anns_valid)
    composite.load(Modes.TRAIN)
    composite.train(anns_train, anns_valid)


if __name__ == '__main__':
    main()
