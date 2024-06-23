#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import cv2
import numpy as np


class Illumination:
    def __init__(self, hsv_range_min=(-0.5, -0.5, -0.5), hsv_range_max=(0.5, 0.5, 0.5)):
        self.hsv_range_min = hsv_range_min
        self.hsv_range_max = hsv_range_max

    def __call__(self, sample):
        # Convert to HSV colorspace from BGR colorspace
        hsv = cv2.cvtColor(sample['img'], cv2.COLOR_RGB2HSV)
        # Generate new random values
        H = 1+np.random.uniform(self.hsv_range_min[0], self.hsv_range_max[0])
        S = 1+np.random.uniform(self.hsv_range_min[1], self.hsv_range_max[1])
        V = 1+np.random.uniform(self.hsv_range_min[2], self.hsv_range_max[2])
        hsv[:, :, 0] = np.clip(H*hsv[:, :, 0], 0, 179)
        hsv[:, :, 1] = np.clip(S*hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(V*hsv[:, :, 2], 0, 255)
        # Convert back to BGR colorspace
        sample['img'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return sample


class HorFlip:
    def __call__(self, sample):
        # Horizontal flipping
        img = sample['img']
        bbox = sample['bbox']
        yaw, pitch, roll = sample['headpose']
        if np.random.uniform(0.0, 1.0) < 0.5:
            sample['img'] = cv2.flip(img, 1)
            sample['bbox'] = np.array([img.shape[1]-bbox[2], bbox[1], img.shape[1]-bbox[0], bbox[3]])
            sample['headpose'] = np.array([-yaw, pitch, -roll])
        return sample


class CropBbox:
    def __init__(self, width, height, bbox_scale):
        self.width = width
        self.height = height
        self.bbox_scale = bbox_scale

    def __call__(self, sample):
        bbox = sample['bbox']
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]
        # Squared bbox required
        max_size = max(bbox_width, bbox_height)
        shift = (float(max_size-bbox_width)/2.0, float(max_size-bbox_height)/2.0)
        bbox_squared = (bbox[0]-shift[0], bbox[1]-shift[1], bbox[2]+shift[0], bbox[3]+shift[1])
        # import copy
        # aux = copy.deepcopy(sample['img'])
        # cv2.rectangle(aux, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0))
        # Enlarge bounding box
        shift = max_size*self.bbox_scale
        bbox_enlarged = (bbox_squared[0]-shift, bbox_squared[1]-shift, bbox_squared[2]+shift, bbox_squared[3]+shift)
        # cv2.rectangle(aux, (int(bbox_enlarged[0]), int(bbox_enlarged[1])), (int(bbox_enlarged[2]), int(bbox_enlarged[3])), (255, 255, 0))
        # cv2.imshow('aa', cv2.cvtColor(aux, cv2.COLOR_BGR2RGB))
        # Project image
        T = np.zeros((2, 3), dtype=float)
        T[0, 0], T[0, 1], T[0, 2] = 1, 0, -bbox_enlarged[0]
        T[1, 0], T[1, 1], T[1, 2] = 0, 1, -bbox_enlarged[1]
        bbox_width = bbox_enlarged[2]-bbox_enlarged[0]
        bbox_height = bbox_enlarged[3]-bbox_enlarged[1]
        S = np.matrix([[self.width/bbox_width, 0, 0], [0, self.height/bbox_height, 0]], dtype=float)
        face_translated = cv2.warpAffine(sample['img'], T, (int(round(bbox_width)), int(round(bbox_height))))
        sample['img'] = cv2.warpAffine(face_translated, S, (self.width, self.height))
        # cv2.imshow('img', cv2.cvtColor(sample['img'], cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        return sample


class ImgPermute:
    def __call__(self, sample):
        # Converts a numpy image in H x W x C format to C x W x H format and changes the range to [0, 1]
        sample['img'] = sample['img'].transpose(2, 0, 1) / 255.0
        return sample
