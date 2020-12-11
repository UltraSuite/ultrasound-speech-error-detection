#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Demo for Error Detection Model described in

Ribeiro, M.S., Cleland, J., Eshky, A., Richmond, K., and Renals. S. (Under Revision)
Automatic detection of speech articulation errors using ultrasound tongue imaging
Submitted to Speech Communication.

Steps applied in this demo are:

# step 1: load input data (MFCCs, ultrasound, normalisers)
# step 2: resize ultrasound and compute speaker mean
# step 3: build input sample
# step 4: ultrasound pre-processing (normalise + append normalised speaker mean)
# step 5: mfcc pre-processing (normalise)
# step 7: forward pass (load model from file and forward pass to get probability distribution)
# step 8: compute final score

This script shows the sequence of steps that are run.
For the details of each step, check the corresponding function in steps.py

Date: 2020
Author: M. Sam Ribeiro
"""

import os
import sys
import steps
import argparse


def main(config):

    ## step 1: load input data

    # MFCCs data (normalised and with deltas)
    mfccs = steps.read_text_array(config['data'], config['file_id']+'.mfccs')

    # raw ultrasound data
    ultrasound, params = steps.read_ultrasound_data(config['data'], config['file_id'])

    # data normalisers
    ultra_mvn   = steps.read_text_array(config['norm'], 'ultrasound.mvn.txt')
    audio_mvn   = steps.read_text_array(config['norm'], 'audio.mvn.txt')
    speaker_mvn = steps.read_text_array(config['norm'], 'speaker.mvn.txt')


    ## step2: resize ultrasound and compute speaker mean
    ultrasound, speaker_mean = steps.preprocess_ultrasound(ultrasound)

    ## step 3: build input sample
    mfcc_sample, ultra_sample = steps.build_sample(mfccs, ultrasound, params, config['timestamp'])

    ## step 4: ultrasound pre-processing
    ultra_sample = steps.normalise_ultrasound(ultra_sample, speaker_mean, \
        normaliser=ultra_mvn, speaker_normaliser=speaker_mvn)

    ## step 5: mfcc pre-processing
    mfcc_sample = steps.normalise_mfcc(mfcc_sample, normaliser=audio_mvn)

    ## step 6: forward pass 
    model_scores = steps.forward_pass(config['model'], mfcc_sample, ultra_sample)

    ## step 7: score
    steps.score_sample(model_scores, config['expected_class'], config['competing_class'], config['threshold'])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fileid',    default='07F-028A', type=str, help='file identifier')
    parser.add_argument('-t', '--timestamp', default=2.83,       type=float, help='timestamp for anchor frame')
    parser.add_argument('-k', '--threshold', default=0.0,        type=float, help='rejection threshold')
    parser.add_argument('-ec', '--expected_class', default='velar',    type=str, help='expected class')
    parser.add_argument('-cc', '--competing_class', default='alveolar', type=str, help='competing class')
    parser.add_argument('-dir', '--base_dir', default='.', type=str, help='base directory')
    args = parser.parse_args()



    # use a single object with all required parameters
    config = {

        # target file identifier
        # we'll look for it (*.ult, *.param, *.mfccs) in the data directory
        'file_id'   : args.fileid,

        # mid-frame of target phone (given by user or discovered automatically)
        # also referred to as 'anchor frame'
        'timestamp' : args.timestamp,

        # competing phone classes. supported clases are:
        # alveolar, dental, labial, labiovelar, lateral, palatal, postalveolar, rhotic, velar
        'expected_class' : args.expected_class,
        'competing_class' : args.competing_class,
        
        # threshold k in the paper (Equation 3, see also Figure 9)
        # higher (positive) values tends to increases recall and decrease precision.
        # lower (negative) values tends to decrease recall and increase precision.
        'threshold'       : args.threshold,

        # input directories with relevant data
        # this script expects to find sub-directories 'data', 'normalisers', and 'model'
        'data'      : os.path.join(args.base_dir, 'sample_data'),
        'norm'      : os.path.join(args.base_dir, 'normalisers'),
        'model'     : os.path.join(args.base_dir, 'model'),
    }

    main(config)

