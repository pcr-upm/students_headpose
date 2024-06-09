#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import copy
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchmetrics.regression import MeanSquaredError
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database
from images_framework.src.composite import Composite
from images_framework.src.viewer import Viewer
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
    Students headpose database script.
    """
    unknown, anns_file = parse_options()
    # Load computer vision components
    composite = Composite()
    sa = StudentsHeadpose('images_framework/alignment/students_headpose/')
    composite.add(sa)

    mean_squared_error = MeanSquaredError()
    composite.parse_options(unknown)
    anns = load_annotations(anns_file)
    composite.load(Modes.TEST)
    preds = []
    target = []
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        for img_pred in pred.images:
            img_pred.clear()
        composite.process(anns[i], pred)
        ann_order = [(img_ann.filename, img_ann.tile) for img_ann in anns[i].images]  # keep order among 'ann' and 'pred'
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, f) & np.array_equal(img_pred.tile, t) for f, t in ann_order].index(True)
            for objs_idx, objs_val in enumerate([img_pred.objects]): #anns[i].images[image_idx].objects, 
                for obj in objs_val:
                    preds.append(obj.headpose.tolist())
            for objs_idx, objs_val in enumerate([anns[i].images[image_idx].objects]):
                for obj in objs_val:
                    target.append(obj.headpose.tolist())

    print(mean_squared_error(torch.tensor(preds), torch.tensor(target)))


if __name__ == '__main__':
    main()
