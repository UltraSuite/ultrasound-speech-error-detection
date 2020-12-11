#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions that run each step for the Error Detection Model.

Date: 2020
Author: M. Sam Ribeiro
"""

import os
import sys
import argparse
import numpy as np
import skimage.transform

import torch

from model import Model



def read_text_array(directory, file_id):
    ''' reads array from text file '''
    data = []
    path = os.path.join(directory, file_id)

    with open(path) as fid:
        for line in fid.readlines():
            row = line.rstrip().split()
            row = list(map(float, row))
            data.append(row)

    data = np.array(data)
    return data


def read_ultrasound_data(directory, target_file_id):
    ''' read ultrasound data and parameters '''

    param_f = os.path.join(directory, target_file_id + '.param')
    ult_f   = os.path.join(directory, target_file_id + '.ult')

    # read ultrasound parameters
    params = {}
    with open(param_f) as param_id:
        for line in param_id:
            name, var = line.partition("=")[::2]
            params[name.strip()] = float(var)

    # read ultrasound data
    fid = open(ult_f, "r")
    ultrasound = np.fromfile(fid, dtype=np.uint8)
    fid.close()

    # convert to float32, which makes computations safer
    ultrasound = ultrasound.astype('float32')
    # reshape to (num_frames, scanlines, pixels)
    ultrasound = ultrasound.reshape((-1, int(params['NumVectors']), int(params['PixPerVector'])))

    return ultrasound, params


def preprocess_ultrasound(sample):
    ''' resizes ultrasound and computes speaker mean '''

    # shape we want to resize ultrasound to: 63 scanlines with 103 pixels
    resize_shape = (63, 103)

    # resize ultrasound (bi-linear interpolation)
    # here, we use skimage's resize function
    # details here: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
    resized = np.empty( (sample.shape[0], resize_shape[0], resize_shape[1]) )

    for i in range(sample.shape[0]):
        frame = sample[i]
        frame = skimage.transform.resize(frame, output_shape=resize_shape, order=1, \
            mode='edge', clip=True, preserve_range=True, anti_aliasing=True)
        resized[i] = frame

    # Compute speaker mean
    # We always append some form of speaker representation to each input sample.
    # The best is probably to compute it over non-silent segments over all speaker data.
    # But we can probably get away with just using utterance mean, which we do here.
    # This is the pixel-wise mean, so dimensions match with input samples.
    # I find that it is better to get the mean after resizing. Rather than before and resize the mean.
    speaker_mean = np.mean(resized, axis=0)

    return resized, speaker_mean


def get_context_indices(anchor, last_frame, window, step):
    '''
        this function finds left and right context windows
        around an anchor frame. See Figure 6 in the paper.
    '''

    # if the time to frame conversion went over the edge
    # we default to the first or last frames
    if anchor < 0: anchor=0
    if anchor > last_frame: anchor = last_frame

    if window == 0:
        indices = np.array([anchor])
    else:
        # indices for left and right context
        left  = np.arange(anchor-window, anchor, step)
        right = np.arange(anchor+step, anchor+window+step, step)
        left, right = left.astype(int), right.astype(int)

        # handle the edges of the stream
        # we simply replicate the first or last frame if we go over
        left     = [max(0, i) for i in left]
        right    = [min(last_frame, i) for i in right]

        # anchor frame comes first in this setup
        # this is for no particular reason, but it shouldn't matter.
        indices = np.array([anchor]+left+right)

    return indices


def build_sample(mfccs, ultrasound, params, timestamp):
    ''' 
        Simplification of the sample build process.
        See Figure 6 of the paper for details.
    '''

    # we're hard-coding these parameters here
    # for ultrasound, they are only applicable to UltraSuite data
    # because we define them based on sample rate
    audio_window, audio_step = 10, 2
    ultra_window, ultra_step = 12, 3

    # ultrasound frame parameters
    frate_ult   = params['FramesPerSec']            # frame rate (in Hz)
    offset_sec  = params['TimeInSecsOfFirstFrame']  # offset with audio (in seconds)
    offset      = offset_sec*frate_ult              # offset with audio (in frames)

    # audio frame rate
    frate_audio = 100. # frame rate audio (in Hz) - assumes 10 ms frames

    # find anchor frames
    audio_anchor_frame = int( round(timestamp * frate_audio) )
    ultra_anchor_frame = int( round( (timestamp*frate_ult) - offset) )

    # get context indices from window and step
    # these are the left and right context windows
    audio_ctx = get_context_indices(audio_anchor_frame, mfccs.shape[0]-1, audio_window, audio_step)
    ult_ctx   = get_context_indices(ultra_anchor_frame, ultrasound.shape[0]-1, ultra_window, ultra_step)

    # retrieve input frames by indices
    mfcc_sample  = mfccs[audio_ctx, :]
    ultra_sample = ultrasound[ult_ctx, :]

    return mfcc_sample, ultra_sample


def normalise_ultrasound(sample, speaker_mean, normaliser, speaker_normaliser):
    ''' normalises ultrasounds and appends speaker mean '''

    # this is the resized shape (63x103)
    # we can infer this from the data or read it from elsewhere
    shape = (sample.shape[1], sample.shape[2])

    # ultrasound normalisers: mean and std
    # normaliser is (2x6489), with row 1 being mean and row 2 std
    mean, std = normaliser
    # here, we reshape to (63x103)
    mean = mean.reshape(shape)
    std  = std.reshape(shape)

    # simple mean-variance normalisation
    # here, we do not bother to tile arrays because numpy broadcasts to the required dimensions
    # if we can't do that, then we want to normalise wach frame separately
    sample = (sample - mean) / std
    sample = sample.astype('float32')

    # normalise speaker mean - same as above
    mean, std = speaker_normaliser
    mean = mean.reshape(shape)
    std  = std.reshape(shape)
    speaker_mean = (speaker_mean - mean) / std
    speaker_mean = speaker_mean.astype('float32')

    # append speaker mean to sample
    # we add a dummy dimension with expand_dims so that we concatenate on the 0th axis
    # e.g. (63x103) -> (1x63x103)
    # then, we append it to the sample (9x63x103)
    speaker_mean = np.expand_dims(speaker_mean, axis=0)
    sample = np.concatenate([sample, speaker_mean], axis=0)

    # we add one more dimension for batch size
    sample = np.expand_dims(sample, axis=0)

    # output shape at this stage should be (1x10x63x103)
    # these are:
    # 1 batch sample
    # 10 input frames (1 anchor frame, 4 left context, 4 right context, 1 spaeker mean)
    # 63 scanlines
    # 103 pixels (resized)
    assert sample.shape == (1, 10, 63, 103)

    return sample


def normalise_mfcc(sample, normaliser):
    ''' Mean variance normalisation for MFCC '''

    # mean and std for normalisation
    # normaliser is (2x60), with row 1 being mean and row 2 std
    mean, std = normaliser
    #  reshape each is (1x60)
    mean = mean.reshape(1, 60)
    std  = std.reshape(1, 60)

    # same as for ultrasound, we do not need to tile
    # because numpy broadcasts to the required dimensions
    sample = (sample - mean) / std
    sample = sample.astype('float32')

    # here, we flatten all frames onto a single vector
    # that is, (11x60) -> (1x660)
    sample = sample.reshape(1, -1)

    # output shape at this stage should be (1x660)
    # this is 
    # 1 batch sample
    # 11 flattened input frames (1 anchor frame, 5 left context, 5 right context)
    # the flattened frames are 11x60 = 660
    assert sample.shape == (1, 660)

    return sample


def forward_pass(model_directory, mfcc, ultra):
    ''' Load pre-trained model from file and do forward pass to get scores '''

    weights_filename = os.path.join(model_directory, 'nnet.pt')

    # originally run in gpu (e.g. cuda:0), but we use cpu now
    device = 'cpu'
    torch.device(device)
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_channels = ultra.shape[1]   # num of ultrasound channels
    mfcc_dim     = mfcc.shape[1]    # num of mffc features
    num_classes  = 9                # num of output classes

    # initialize model, load pre-trained weights from file, and set eval mode
    model = Model(num_channels, mfcc_dim, num_classes)
    model.load_state_dict(torch.load(weights_filename))
    model.eval()

    # convert data to torch tensors
    mfcc  = torch.from_numpy(mfcc).float()
    ultra = torch.from_numpy(ultra).float()

    # forward pass - this calls model.forward()
    scores = model(ultra, mfcc)
    # here, we detach from the torch graph and store data as a list 
    scores = scores.detach().numpy().tolist()[0]

    return scores


def score_sample(model_scores, expected_class_y, competing_class_c, threshold):
    ''' final scoring, given probability distribution given by the model '''


    # this is the ordered set of labels used by the classifier
    # model_scores represents logp for each of these classes
    label_set = (\
        'alveolar', 'dental', 'labial', 'labiovelar', \
        'lateral', 'palatal', 'postalveolar', 'rhotic', 'velar'\
        )

    # find the indices of expected and competing classes
    y_index = label_set.index(expected_class_y)
    c_index = label_set.index(competing_class_c)
    # and get the model score
    # here, s_y and s_c represent logp(y|x) and logp(c|x) in the paper
    s_y  = model_scores[y_index]
    s_c  = model_scores[c_index]

    # Equations 1 and 3, respectively
    s_m = s_y - s_c
    b_s = 0 if s_m > threshold else 1
    print('Score for {0}-{1} substitution: {2:.4f}. Error: {3}'.\
        format(expected_class_y, competing_class_c, s_m, bool(b_s)))

    # An example on how to score without a competing class c
    # This is the same as Equation 2
    s_c = max([q for i, q in enumerate(model_scores) if i!=y_index])
    # Equations 1 and 3, as before
    s_m = s_y - s_c
    b_s = 0 if s_m > threshold else 1
    print('Score for generic {0} production: {1:.4f}. Error: {2}'\
        .format(expected_class_y, s_m, bool(b_s)))

    # Note: b_s is either 0 (expected class) or 1 (competing class)
    # Alternatively, we may use s_m directly, which will include some idea of confidence
    # For example, we can use s_m to rank samples
