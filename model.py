#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model architecture for PyTorch

Date: 2020
Author: M. Sam Ribeiro
"""

import numpy as np
np.random.seed(42)

import random
random.seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):

    def __init__(self, num_channels, audio_dim, num_classes):
        super(Model, self).__init__()

        # Encoder size is the flattened features from the ultrasound encoder.
        # we normally estimate this based on input dimensions, number of
        # channels, or kernel size. Since this is a pre-trained model with
        # fixed-sized inputs, we hard-code it here for simplicity.
        self.encoder_size = 16896

        # Audio Encoder
        self.audio_fc1 = nn.Linear(audio_dim, 256, bias=True)

        # Ultrasound Encoder
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.batch_norm = nn.BatchNorm1d(self.encoder_size+256)

        # phone classifier
        self.fc1 = nn.Linear(self.encoder_size+256, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, num_classes, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, ultra, audio):
        ''' forward pass '''
        u = ultra
        a = audio

        # encode audio
        a = F.relu( self.audio_fc1(a) )

        # encode ultrasound
        u = F.max_pool2d(F.relu(self.conv1(u)), kernel_size=(2, 2))
        u = F.max_pool2d(F.relu(self.conv2(u)), kernel_size=(2, 2))
        u = u.view(-1, self.encoder_size)  # flatten

        # join features and normalise
        x = torch.cat([u, a], dim=1)
        x = self.batch_norm(x)

        # phone classifier
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = self.fc3(x)
        x = self.softmax(x)

        return x
