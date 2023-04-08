# Semantic-based Implicit Feature Transform for Few-Shot Classification

PyTorch implementation of Semantic-based Implicit Feature Transform for Few-Shot Classification


![](framework.png)



## Dependencies
* python 3.6.5
* numpy 1.16.0
* torch 1.8.0
* tqdm 4.57.0
* scipy 1.5.4
* torchvision 0.9.0

## Datasets

## Backbone
**WRN-28-10:** download the pretrained backbones from here: [checkpoints](https://drive.google.com/drive/folders/1KfPzwMvVzybvp13IQW5ipHvSxBncTA-C)   
**ResNet12:** download the pretrained backbones from here: [checkpoints](https://drive.google.com/file/d/1Prn7_41NVrZbnePAlSiKjD21Jlz0LKJM/view)


## Evaluate our SIFT method

To evaluate our SIFT method, run:
```eval
python3 run_st.py
```

## Acknowlegements
[S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)
[DC](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)
