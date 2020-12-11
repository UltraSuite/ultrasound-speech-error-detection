# Exploiting ultrasound tongue imaging for the automatic detection of speech articulation errors


This repository provides a demo for the set of experiments described in the paper:

Ribeiro, M.S., Cleland, J., Eshky, A., Richmond, K., and Renals. S. (Under Revision). **Exploiting ultrasound tongue imaging for the automatic detection of speech articulation errors**. Submitted to Speech Communication.


#### Requirements

Please make sure the following Python libraries and their dependencies are available.

- numpy (1.16.0)
- scikit-image (0.17.2)
- pytorch (1.3.1)

#### Usage

There is some sample data available in `sample_data`. To run the demo, type:

`python run_demo.py`

This will run the demo with all default parameters. You can optionally configure them. The example below will evaluate the segment at 4.07 seconds (an alveolar) for velar substitution.

`python run_demo.py --timestamp 4.07 --expected_class alveolar --competing_class velar`

Alternatively, you can test the same segment for velar fronting, which should be identified as an error.

`python run_demo.py --timestamp 4.07 --expected_class velar --competing_class alveolar`

You can test more utterances using ultrasound and audio. There are plenty available from the [Ultrasuite Repository](https://ultrasuite.github.io/), with manually verified word boundaries and with the scores given by the Speech and Language Therapists described in the paper. If you use different utterances, you will need to generate MFFCs for them.

#### Audio feature extraction 

The audio stream inputs MFCCs. These can be computed using any publicly available library. We have used Kaldi to generate these features. For the sample utterance in `sample_data`, we provide a set of MFFCs. In the directory `make_mfccs_demo`, there is a example of the steps we used to extract those features. If you wish to run `make_mfccs.sh`, you will need Kaldi installed on your system.

#### Pre-trained model

The pre-trained model parameters available in `models` is the system using both ultrasound and audio and trained on the joint [TaL corpus](https://ultrasuite.github.io/data/tal_corpus/) and [UXTD dataset](https://ultrasuite.github.io/data/uxtd/). This model has 86.90% accuracy when evaluated on Typically Developing Data. See Section 5 of the paper for more details.

#### Accepted classes

The classes that the model allows are the following: 

*alveolar, dental, labial, labiovelar, lateral, palatal, postalveolar, rhotic, velar*. 

You can obtain generic scores for productions of these classes or test for any substitution between them.

#### Notes

This repository is designed to demo the set of steps used to compute phone scores. It is heavily commented to illustrate what is expected at each stage, from preprocessing, normalisation, to forward pass.

This repository is not designed to replicate model training or evaluation. Therefore we made some simplifications in the demo. For example, we run the model on CPU rather than GPU, we hard-code some paramters, and we remove some for-loops and batch processing code used to handle multiple samples.
