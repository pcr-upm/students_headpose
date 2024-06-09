#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import copy
from tqdm import tqdm
from pathlib import Path
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
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    parser.add_argument('--save-file', '-f', dest='save_file', action="store_true",
                        help='Save experiments in a text file.')
    parser.add_argument('--save-image', '-i', dest='save_image', action="store_true",
                        help='Save processed images.')
    args, unknown = parser.parse_known_args()
    anns_file = args.anns_file
    show_viewer = args.show_viewer
    save_file = args.save_file
    save_image = args.save_image
    return unknown, anns_file, show_viewer, save_file, save_image


def main():
    """
    Students headpose database script.
    """
    unknown, anns_file, show_viewer, save_file, save_image = parse_options()
    # Load computer vision components
    composite = Composite()
    sa = StudentsHeadpose('images_framework/alignment/students_headpose/')
    composite.add(sa)

    composite.parse_options(unknown)
    anns = load_annotations(anns_file)
    composite.load(Modes.TEST)
    if show_viewer:
        viewer = Viewer('images_viewer')
    if save_file:
        ofs = open('images_framework/output/results.txt', 'w', encoding='utf-8')
    if save_image:
        viewer = Viewer('images_save')
        dirname = 'images_framework/output/images/'
        Path(dirname).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        for img_pred in pred.images:
            img_pred.clear()
        composite.process(anns[i], pred)
        if show_viewer:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.show(0)
        if save_file:
            composite.evaluate(ofs, anns[i], pred)
        if save_image:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.save(dirname)
            composite.save(dirname, pred)
    if save_file:
        ofs.close()


if __name__ == '__main__':
    main()
