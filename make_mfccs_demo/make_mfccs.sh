#!/bin/bash

# this script is a simplification of the feature extraction process for the audio stream.

# for simplicity, we run each command separately (rather than piping)
# the 't' flag (e.g. 'ark,t') tells Kaldi to store data as text files

# this process is non-deterministic, so you might notice slight discrepancies in the output
# non-deterministic behaviour is caused by dithering in 'sox' and 'compute-mfcc-feats'
# disable dithering in both tools to produce deterministic output


# extract raw MFCCs features
compute-mfcc-feats --verbose=2 --config=./mfcc.conf scp:./wav.scp ark,t:output/raw.mfccs.txt

# compute stats for cepstral mean variance normalization
compute-cmvn-stats ark:output/raw.mfccs.txt ark:output/cmvn.txt

# normalise
apply-cmvn ark:output/cmvn.txt ark:output/raw.mfccs.txt ark,t:output/norm.mfccs.txt

# add delta and delta-delta features
add-deltas ark:output/norm.mfccs.txt ark,t:output/deltas.mfccs.txt

# the file output/deltas.mfccs.txt should be the equivalent of 07F-028A.mfccs
# you'll note that the output of these commands uses the format:
# "utterance-identifier [matrix]" to store matrices for multiple files
# this can be used when processing multiple utterances
# the input to the demo run.py simplifies this by stripping utterance identifiers
# you can rewrite the read_mfccs function to allow Kaldi-style format
