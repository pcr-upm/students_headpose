#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD, lr_scheduler, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitResNet(pl.LightningModule):
    """
    Define a new class to turn the ResNet model that we want to use as a feature extractor.
    """
    resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152}

    def __init__(self, num_classes, version, loss_calculator, lr=1e-3, patience=20, batch_size=16, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        # Using a pretrained ResNet backbone
        self.model = self.resnets[version](pretrained=transfer)
        # Replace old FC layer with Identity, so we can train our own
        linear_size = list(self.model.children())[-1].in_features
        # Replace final layer for fine-tuning
        self.model.fc = nn.Linear(in_features=linear_size, out_features=num_classes)
        self.loss_calculator = loss_calculator
        # Option to only tune the fully-connected layers
        if tune_fc_only:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8
        )

        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-3,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def _step(self, batch):
        inputs = batch['img'].float()
        targets = batch['headpose'].float()
        outputs = self.model(inputs)
        return self.loss_calculator.calculate_loss(outputs, targets)

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
