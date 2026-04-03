# Implementation of LLM-proposed Open Taxonomy Described in "Your thoughts tell who you are..."

This repo contains an implementation of the LLM-proposed open taxonomy (LOT) algorithm described in the preprint "Your thoughts tell who you are: Characterize the reasoning patterns of LRMs" (https://arxiv.org/abs/2509.24147)

Currently under construction


## Dependencies 
Install the required dependencies by `conda env create --name reasoning --file=environment_ymls/countdown-full-environment.yml`

## Table of Contents
- `coder_por_full_training_procedure.py` is the Python script for training LOT to generate a taxonomy that describes the unique reasoning behaviors that distinguish two models' reasoning traces on solving the same questions. PoR stands for presence of reasoning. In this script, the LOT encodes a reasoning trace into a binary vector where each dimension corresponding to a distinguishing reasoning behavior and the value indicating whether the specific behavior is observed in the trace or not. A linear logistic regression classifier will be trained on the encodings to predict their source LLMs.
- `coder_bor_full_training_procedure.py` is same as above but encodes the reasoning traces using bag of reasonings vectors which describe both the presence of a reasoning behavior and how many times that the reasoning behavior occurred in the traces.


## BibTeX
> @article{chen2025your,
  title={Your thoughts tell who you are: Characterize the reasoning patterns of LRMs},
  author={Chen, Yida and Mao, Yuning and Yang, Xianjun and Ge, Suyu and Bi, Shengjie and Liu, Lijuan and Hosseini, Saghar and Tan, Liang and Nie, Yixin and Nie, Shaoliang},
  journal={arXiv preprint arXiv:2509.24147},
  year={2025}
}