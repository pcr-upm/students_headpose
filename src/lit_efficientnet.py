#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Iñigo Sanz'
__email__ = 'inigo.sanz.torres@alumnos.upm.es'

import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch.optim import SGD, Adam
from torchmetrics.regression import MeanAbsoluteError


class LitEfficientNet(pl.LightningModule):
    """
    Define a new class to turn the EfficientNet model that we want to use as a feature extractor.
    """
    efficientnets = {0: models.efficientnet_v2_s}
    optimizers = {'adam': Adam, 'sgd': SGD}

    def __init__(self, num_classes, version, optimizer='adam', lr=1e-3, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = self.optimizers[optimizer]
        # Loss criterion
        self.loss_fn = nn.MSELoss()
        # MAE metric
        self.mae = MeanAbsoluteError()
        # Using a pretrained backbone
        self.model = self.efficientnets[version](pretrained=transfer)
        # Replace old FC layer with Identity, so we can train our own
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        # Option to only tune the fully-connected layers
        if tune_fc_only:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        inputs = batch['img'].float()
        targets = batch['headpose'].float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        mae = self.mae(outputs, targets)
        return loss, mae

    def training_step(self, batch, batch_idx):
        loss, mae = self._step(batch)
        # Perform logging
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log('train_mae', mae, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae = self._step(batch)
        # Perform logging
        self.log('val_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log('val_mae', mae, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, mae = self._step(batch)
        # Perform logging
        self.log('test_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log('test_mae', mae, batch_size=self.batch_size, on_step=False, on_epoch=True)
