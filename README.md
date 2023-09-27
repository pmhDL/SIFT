# Semantic-based Implicit Feature Transform for Few-Shot Classification

PyTorch implementation of Semantic-based Implicit Feature Transform for Few-Shot Classification

## Dependencies
* python 3.6.5
* numpy 1.16.0
* torch 1.8.0
* tqdm 4.57.0
* scipy 1.5.4
* torchvision 0.9.0

## Overview
Few-shot learning aims to recognize instances from previously unseen classes based on a very limited number of examples. However, models often face the challenge of overfitting due to the biased distribution computed from extremely scarce training data. This work proposes a Semantic-based Implicit Feature Transform (SIFT) method to implicitly generate high-quality features for few-shot learning tasks. In this method, we employ an encode-transform-decode pipeline to facilitate the direct transfer of feature instances from base classes to novel classes via semantic transformation, ensuring the generation of semantically meaningful features. Additionally, a compactness constraint is imposed on the generated features, with novel class prototypes serving as anchors, to ensure that the features are distributed around the class centers. Furthermore, we integrate the cluster centers of the query set with the initial prototypes computed from the support set to produce less biased class prototypes, which can serve as better anchors for feature reconstruction. The experimental results reveal that our method outperforms the baselines by substantial margins and achieves state-of-the-art few-shot classification performance on the miniImageNet and CIFAR-FS datasets in both inductive and transductive settings, demonstrating the superiority of our method. Furthermore, the comprehensive ablation studies provide additional validation of its effectiveness.

![Image text](https://github.com/pmhDL/SIFT/blob/main/Image/framework.png)

## Datasets
The dataset can be downloaded from the following links:
* [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
* [CUB](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)
* [glove word embedding](https://nlp.stanford.edu/projects/glove/)

## Pre-trained Backbones
The pre-trained backbones can be downloaded from the following links:
* [WRN-28-10](https://drive.google.com/drive/folders/1KfPzwMvVzybvp13IQW5ipHvSxBncTA-C)   
* [ResNet12](https://drive.google.com/drive/folders/1unnbnYgjXtwP4lFtcLrCAcZ_H1uQESLf)

## Evaluate SIFT method
To evaluate SIFT, run:
```eval
python3 run_st.py
```

## LISENCE
* All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode
* The license gives permission for academic use only.

## Acknowlegements
Our project references the codes in the following repos.
* [S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)
* [DC](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)
