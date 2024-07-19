#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitEfficientNet(pl.LightningModule):
    """
    Define a new class to turn the EfficientNet model that we want to use as a feature extractor.
    """
    efficientnets = {0: models.efficientnet_b4}

    def __init__(self, num_classes, version, lr=1e-3, patience=20, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        # Loss criterion
        self.loss_fn = nn.L1Loss()  # MAE metric
        # Using a pretrained EfficientNet backbone
        self.model = self.efficientnets[version](weights='IMAGENET1K_V1' if transfer else None)
        # Replace old FC layer with Identity, so we can train our own
        linear_size = self.model.classifier[1].in_features
        # Replace final layer for fine-tuning
        self.model.classifier[1] = nn.Linear(in_features=linear_size, out_features=num_classes)
        # Option to only tune the fully-connected layers
        if tune_fc_only:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=int(round(self.patience/4)))
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def _step(self, batch):
        inputs = batch['img'].float()
        targets = batch['headpose'].float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('val_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        # Perform logging
        self.log('learning_rate', lr, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        # Perform logging
        self.log('test_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
