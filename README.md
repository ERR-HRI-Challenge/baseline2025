# ERR @ HRI 2.0

Website: [https://sites.google.com/cam.ac.uk/err-hri](https://sites.google.com/view/errhri20)

The ERR@HRI challenge aims at addressing the problem of failure detection in human-robot interaction (HRI) by providing the community with the means to benchmark efforts for mono-modal vs. multi-modal robot failure detection in HRI. 

## Contents 
### Dataset 

[data_pre_processing_rf_train](./dataset/data_pre_processing_rf_train.py) conatins the baseline code used for pre-processing the training data. 


### Training

[rf_baseline_challenge1](./training/rf_baseline_challenge1.ipynb) contains the code for training the baseline for sub-challenge 1. 
[rf_baseline_challenge1](./training/rf_baseline_challenge2.ipynb) contains the code for training the baseline for sub-challenge 2. 

### Evaluation

[challenge1_eval](./evaluation/challenge1_eval.ipynb) contains the code for evaluating predictions for sub-challenge 1. 
[challenge2_eval](./evaluation/challenge2_eval.ipynb) contains the code for evaluating predictions for sub-challenge 2. 


### Baseline models

The final baseline models can be found in [baseline_models](./baseline_models.zip)


### Evaluation

The submitted models, for each task, will be evaluated on **error detection rate**, **number of false positives**, and **overall F1 score**. 

#### Overall performance

We will rank models based on the overall F1-score.

### For more details about the challenge, dataset, and baseline, please refer to the manuscript. 
