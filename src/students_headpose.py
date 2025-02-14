#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from images_framework.alignment.students_headpose.src.dataloader import MyDataset, Mode
from images_framework.alignment.students_headpose.src.losses import GeodesicLoss
from images_framework.alignment.students_headpose.src.pcrlogger import PCRLogger
from images_framework.src.alignment import Alignment
from images_framework.alignment.students_headpose.src.pose_representations import PoseRepresentation, PoseRepresentationFactory, PoseLossCalculator, PoseLossCalculatorFactory, EulerPose, L1LossCalculator, ConversionParams

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class Backbone(Enum):
    RESNET = 'resnet'
    EFFICIENTNET = 'efficientnet'


class StudentsHeadpose(Alignment):
    """
    Head pose estimation using a popular algorithm
    """

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpus = None
        self.device = None
        self.backbone = None
        self.version = None
        self.batch_size = None
        self.epochs = None
        self.patience = None
        self.order = None
        self.width = 256
        self.height = 256
        self.loss_calculator = None
        self.pose = None
        self.bg_dir = None

    def parse_options(self, params):
        unknown = super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='StudentsHeadpose', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--backbone', dest='backbone', required=True, choices=[x.value for x in Backbone],
                            help='Select backbone model.')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                            help='Number of images in each mini-batch.')
        parser.add_argument('--epochs', dest='epochs', type=int, default=200,
                            help='Number of sweeps over the dataset to train.')
        parser.add_argument('--patience', dest='patience', type=int, default=20,
                            help='Number of epochs with no improvement after which training will be stopped.')
        parser.add_argument('--pose', type=str, choices=[x.name for x in PoseRepresentation.__subclasses__()],
                            default=EulerPose.name,
                            help=f'Internal pose parameterization of the network (default: {EulerPose.name}).')
        parser.add_argument('--loss', type=str, choices=[x.name for x in PoseLossCalculator.__subclasses__()],
                            default=L1LossCalculator.name,
                            help=f'Loss function (default {L1LossCalculator.name}.')
        parser.add_argument('--bg-dir', dest='bg_dir', required=False,
                            help='Background substitution directory.', default=None)
        args, unknown = parser.parse_known_args(unknown)
        print(parser.format_usage())
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.gpus = args.gpu
        self.device = torch.device('cuda' if mode_gpu else 'cpu')
        self.backbone = Backbone(args.backbone)
        self.version = 50 if self.backbone is Backbone.RESNET else 0
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.patience = args.patience
        if self.database in ['aflw', 'op3d12p', 'dad', 'all']:
            self.order = 'YXZ'
        elif self.database in ['300wlp', 'panoptic', 'aflw2000']:
            self.order = 'XYZ'
        else:
            raise ValueError('Database is not implemented')
        params = ConversionParams(order=self.order, device=self.device)
        self.pose = PoseRepresentationFactory.create_pose_representation(args.pose, params)
        pose_target = PoseRepresentationFactory.create_pose_representation(EulerPose.name, params)
        self.loss_calculator = PoseLossCalculatorFactory.create_loss_calculator(args.loss, self.pose, pose_target)
        self.bg_dir = args.bg_dir

    def train(self, anns_train, anns_valid):
        import pytorch_lightning as pl
        from pytorch_lightning import loggers as pl_loggers
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        # Prepare dataloaders
        dataset_train = MyDataset(anns_train, self.order, self.width, self.height, Mode.TRAIN, bg_train_folder=self.bg_dir)
        dataset_valid = MyDataset(anns_valid, self.order, self.width, self.height, Mode.VALID)
        drop_last = (len(dataset_train) % self.batch_size) == 1  # discard a last iteration with a single sample
        dl_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=drop_last)
        dl_valid = DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                              drop_last=False)
        # Train the model
        print('Train model')
        accelerator = 'gpu' if 'cuda' in str(self.device) else 'cpu'
        model_path = self.path + 'data/' + self.database + '/' + self.backbone.value + '/' + self.pose.name + '/' + self.loss_calculator.name + '/'
        ckpt_path = os.path.join(model_path + 'ckpt/', 'last.ckpt')
        loggers = [pl_loggers.TensorBoardLogger(save_dir=model_path + 'logs/', default_hp_metric=False), PCRLogger()]
        early_callback = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)
        ckpt_callback = ModelCheckpoint(dirpath=model_path + 'ckpt/', filename='{epoch}-{val_loss:.5f}',
                                        monitor='val_loss', save_last=True, save_top_k=1)
        trainer = pl.Trainer(accelerator=accelerator, devices=self.gpus, enable_progress_bar=False,
                             max_epochs=self.epochs, precision=32, deterministic=True, gradient_clip_val=None,
                             logger=loggers, callbacks=[early_callback, ckpt_callback])
        trainer.fit(model=self.model, train_dataloaders=dl_train, val_dataloaders=dl_valid,
                    ckpt_path=ckpt_path if os.path.isfile(ckpt_path) else None)

    def load(self, mode):
        import torchinfo
        from images_framework.src.constants import Modes
        from images_framework.alignment.students_headpose.src.lit_resnet import LitResNet
        from images_framework.alignment.students_headpose.src.lit_efficientnet import LitEfficientNet
        # Set up the neural network to train
        print('Load model')
        torch.set_float32_matmul_precision('medium')
        if self.backbone is Backbone.RESNET:
            self.model = LitResNet(num_classes=self.pose.num_classes, version=self.version, lr=1e-4,
                                   patience=self.patience, batch_size=self.batch_size, transfer=True,
                                   tune_fc_only=False, loss_calculator=self.loss_calculator)
        elif self.backbone is Backbone.EFFICIENTNET:
            self.model = LitEfficientNet(num_classes=self.pose.num_classes, version=self.version, lr=1e-3,
                                         patience=self.patience, batch_size=self.batch_size, transfer=True,
                                         tune_fc_only=False, loss_calculator=self.loss_calculator)
        else:
            raise ValueError('Backbone is not implemented')
        self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, 3, self.width, self.height), device=self.device.type,
                          col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
        # Set up the neural network to test
        if mode is Modes.TEST:
            model_path = self.path + 'data/' + self.database + '/' + self.backbone.value + '/' + self.pose.name + '/' + self.loss_calculator.name + '/'
            print('Loading model from {}'.format(model_path))
            if self.backbone is Backbone.RESNET:
                self.model = LitResNet.load_from_checkpoint(os.path.join(model_path + 'ckpt/', 'best.ckpt'),
                                                            num_classes=self.pose.num_classes, version=self.version, loss_calculator=self.loss_calculator)
            elif self.backbone is Backbone.EFFICIENTNET:
                self.model = LitEfficientNet.load_from_checkpoint(os.path.join(model_path + 'ckpt/', 'best.ckpt'),
                                                                  num_classes=self.pose.num_classes,
                                                                  version=self.version, loss_calculator=self.loss_calculator)
            self.model.eval()

    def process(self, ann, pred):
        from scipy.spatial.transform import Rotation
        # Prepare dataloader
        dataset_test = MyDataset([pred], self.order, self.width, self.height, Mode.TEST)
        dl_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             drop_last=False)
        with torch.no_grad():
            for batch in dl_test:
                # Generate prediction
                output = self.model(batch['img'].float().to(self.device))
                # Save prediction
                obj_pred = pred.images[batch['idx_img']].objects[batch['idx_obj']]
                obj_pred.headpose = self.pose.convert_to_rotation_matrix(output)
