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
Few-shot learning (FSL) as a data-scarce method, aims to recognize instances of unseen classes solely based on very few examples. However, the model can easily become overfitted due to the biased distribution formed with extremely limited training data. This paper presents a task specific data augmentation approach based on semantic transformation by implicitly transferring samples from base dataset to the novel tasks, which guarantees generating semantically meaningful features. Specifically, the feature transfer process is carried out in semantic space. We further impose a compactness constraint to the generated features with the prototypes working as the reference points, which ensures the generated features distribute around the class centers. Moreover, we incorporate the cluster centers of the query set with the prototypes of the support set to reduce the bias of the class centers. With the supervision of the compactness loss, the model is encouraged to generate discriminative features with high inter-class dispersion and intra-class compactness. Extensive experiments show that our method outperforms the state-of-the-arts on four benchmarks, namely MiniImageNet, TieredImageNet, CUB and CIFAR-FS. 
![Image text](https://github.com/pmhDL/SIFT/blob/main/Images/framework.png)

## Datasets
* [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
* [CUB](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)
* [glove word embedding](https://nlp.stanford.edu/projects/glove/)

## Pre-trained Backbones
The pre-trained backbones can be downloaded from the following links:
* [WRN-28-10](https://drive.google.com/drive/folders/1KfPzwMvVzybvp13IQW5ipHvSxBncTA-C)   
* [ResNet12](https://drive.google.com/file/d/1Prn7_41NVrZbnePAlSiKjD21Jlz0LKJM/view)

## Evaluate SIFT method
To evaluate SIFT, run:
```eval
python3 run_st.py
```

## LISENCE
* All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode
* The license gives permission for academic use only.

## Acknowlegements
* [S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)
* [DC](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)
