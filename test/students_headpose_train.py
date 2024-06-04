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


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns-file', '-a', dest='anns_file', required=True,
                        help='Ground truth annotations file.')
    args, unknown = parser.parse_known_args()
    anns_file = args.anns_file
    return unknown, anns_file

def main():
    """
    Students headpose train script.
    """
    unknown, anns_file = parse_options()
    # Load computer vision components
    composite = Composite()
    sa = StudentsHeadpose('images_framework/alignment/students_headpose/')
    composite.add(sa)

    composite.parse_options(unknown)
    anns = load_annotations(anns_file)
    composite.load(Modes.TRAIN)
    anns_train, anns_valid = train_test_split(anns, test_size=0.2, random_state=1, shuffle=True)
    composite.train(anns_train, anns_valid)


if __name__ == '__main__':
    main()