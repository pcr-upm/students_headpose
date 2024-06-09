#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from images_framework.src.alignment import Alignment
from images_framework.alignment.students_headpose.src.pcrlogger import PCRLogger
from images_framework.alignment.students_headpose.src.dataloader import MyDataset, Mode
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class StudentsHeadpose(Alignment):
    """
    Head pose estimation using a popular algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.device = None
        self.width = 224
        self.height = 224

    def parse_options(self, params):
        unknown = super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='StudentsHeadpose', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                            help='Number of images in each mini-batch.')
        parser.add_argument('--epochs', dest='epochs', type=int, default=200,
                            help='Number of sweeps over the dataset to train.')
        parser.add_argument('--patience', dest='patience', type=int, default=10,
                            help='Number of epochs with no improvement after which training will be stopped.')
        args, unknown = parser.parse_known_args(unknown)
        print(parser.format_usage())
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.device = torch.device('cuda:{}'.format(args.gpu[0]) if mode_gpu else 'cpu')
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.patience = args.patience

    def train(self, anns_train, anns_valid):
        import pytorch_lightning as pl
        from pytorch_lightning import loggers as pl_loggers
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        # Prepare dataloaders
        dataset_train = MyDataset(anns_train, self.database, self.width, self.height, Mode.TRAIN)
        dataset_valid = MyDataset(anns_valid, self.database, self.width, self.height, Mode.VALID)
        drop_last = (len(dataset_train) % self.batch_size) == 1  # discard a last iteration with a single sample
        dl_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
        dl_valid = DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        # Train the model
        print('Train model')
        model_path = self.path + 'data/' + self.database + '/'
        ckpt_path = os.path.join(model_path+'ckpt/', 'last.ckpt')
        loggers = [pl_loggers.TensorBoardLogger(save_dir=model_path+'logs/'), PCRLogger()]
        checkpoint_callback = ModelCheckpoint(dirpath=model_path+'ckpt/', filename='{epoch}-{val_loss:.5f}', monitor='val_loss', save_last=True, save_top_k=1)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)
        trainer = pl.Trainer(logger=loggers, accelerator='auto', devices='auto', enable_progress_bar=True, max_epochs=self.epochs, precision=32, deterministic=True, gradient_clip_val=None, callbacks=[checkpoint_callback, early_stopping])
        trainer.fit(model=self.model, train_dataloaders=dl_train, val_dataloaders=dl_valid, ckpt_path=ckpt_path if os.path.isfile(ckpt_path) else None)

    def load(self, mode):
        import torchsummary
        from images_framework.src.constants import Modes
        from images_framework.alignment.students_headpose.src.resnet_classifier import ResNetClassifier
        from images_framework.alignment.students_headpose.src.efficientnet_classifier import EfficientnetClassifier
        # Set up the neural network to train
        print('Load model')
        torch.set_float32_matmul_precision('medium')
        self.model = ResNetClassifier(num_classes=3, resnet_version=18, optimizer='adam', lr=1e-4, batch_size=self.batch_size, transfer=True, tune_fc_only=False)
        torchsummary.summary(self.model, input_size=(3, self.width, self.height), batch_size=self.batch_size, device='cpu')
        # Set up the neural network to test
        if mode is Modes.TEST:
            model_path = self.path + 'data/' + self.database + '/'
            print('Loading model from {}'.format(model_path))
            self.model = ResNetClassifier.load_from_checkpoint(os.path.join(model_path+'ckpt/', 'best.ckpt'), num_classes=3, resnet_version=18)
            self.model.to(self.device)
            self.model.eval()

    def process(self, ann, pred):
        from scipy.spatial.transform import Rotation
        # Prepare dataloader
        dataset_test = MyDataset([pred], self.database, self.width, self.height, Mode.TEST)
        dl_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        with torch.no_grad():
            for batch in dl_test:
                # Generate prediction
                euler = self.model(batch['img'].float().to(self.device)).squeeze().cpu().numpy()
                # Save prediction
                obj_pred = pred.images[batch['idx_img']].objects[batch['idx_obj']]
                obj_pred.headpose = Rotation.from_euler('YXZ', euler, degrees=True).as_matrix()
