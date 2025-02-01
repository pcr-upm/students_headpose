#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os.path

import cv2
import numpy as np
import random
from itertools import cycle
import secrets
from enum import Enum
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation
from images_framework.alignment.students_headpose.src.transformations import Illumination, HorFlip, SimTform, CropBbox, ImgPermute, BgSubstitution


class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class MyDataset(Dataset):
    """
    Create a dataset class for our head pose estimation data sets.
    """
    def __init__(self, anns, order, width, height, mode: Mode, bg_train_file=None):
        self.order = order
        self.width = width
        self.height = height
        self.mode = mode
        # Set data information
        self.img_indices, self.obj_indices, self.filepaths, self.bboxes, self.headpose = [], [], [], [], []
        self.bg_train_file = bg_train_file
        self.bg_transformation = []
        if self.bg_train_file:
            self.__setup_bg_images()
        for ann in anns:
            for img_idx, img_ann in enumerate(ann.images):
                for obj_idx, obj_ann in enumerate(img_ann.objects):
                    self.img_indices.append(img_idx)
                    self.obj_indices.append(obj_idx)
                    self.filepaths.append(img_ann.filename)
                    self.bboxes.append(np.array(obj_ann.bb, dtype=np.float64))
                    self.headpose.append(obj_ann.headpose)

    def __len__(self):
        # Returns the length of the dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load image
        # This is memory efficient because all the images are not stored in the memory at once but read as required
        # TODO load matting mask and random bg image if flag is set
        image = cv2.imread(self.filepaths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        euler = Rotation.from_matrix(self.headpose[idx]).as_euler(self.order, degrees=True)
        sample = {'filepath': self.filepaths[idx], 'img': image, 'idx_img': self.img_indices[idx], 'idx_obj': self.obj_indices[idx], 'bbox': self.bboxes[idx], 'headpose': euler}
        # Composes several transforms together
        if self.mode is Mode.TRAIN:
            ops = self.bg_transformation + [Illumination((0.1, 0.2, 0.2)), HorFlip(self.order), SimTform(self.order, 5, 0.2, 0.1), CropBbox(self.width, self.height, 0.3), ImgPermute()]
        else:
            ops = [CropBbox(self.width, self.height, 0.3), ImgPermute()]
        sample = transforms.Compose(ops)(sample)
        return sample

    def __setup_bg_images(self):
        bg_parent_folder = os.path.dirname(self.bg_train_file)
        bg_parent_folder = os.path.join(bg_parent_folder, 'train')
        try:
            with open(os.path.join(self.bg_train_file), 'r', encoding='utf-8') as f:
                self.bg_image_file_names = [
                    os.path.join(bg_parent_folder, line.strip()) for line in f.readlines()
                    if line.strip() and not line.startswith('#')
                ]
                if not self.bg_image_file_names:
                    print(f"Warning: File {self.bg_train_file} is empty")

        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.bg_train_file} not found") from None
        except UnicodeDecodeError:
            raise IOError(f"Codification error reading file {self.bg_train_file}") from None
        except Exception as e:
            raise IOError(f"Unexpected error reading file {self.bg_train_file}: {str(e)}") from None

        self.seed = secrets.randbelow(2**32)
        print('Using seed:', self.seed)
        random.seed(self.seed)
        random.shuffle(self.bg_image_file_names)
        self.bg_image_file_names = cycle(self.bg_image_file_names)
        self.bg_transformation = [BgSubstitution(self.bg_image_file_names)]