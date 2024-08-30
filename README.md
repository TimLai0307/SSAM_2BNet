# SSAM_2BNet

# Environment
- Python 3.8
- pytorch 1.10.2

Please run the follow line to install enviroment
```python

pip install -r requirements.txt

```

# How to try

## Download dataset (Places365、CelebA、ImageNet)
[ShanghaiTech](https://www.kaggle.com/datasets/tthien/shanghaitech)  (no official)

[UCF_CC_50](https://www.crcv.ucf.edu/data/ucf-cc-50/)

[UCF_QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/)

[NWPU](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)

[JHU_CROWD++](http://www.crowd-counting.com/#download)

## Data preprocess

Edit the root and cls in generate_density.py
```python

root = 'The root of image'
target_root = 'The root of saving generate ground-truth'
cls = 'For which dataset' # Ex. SHH, NWPU, UCF_QNRF, UCF_CC_50, jhu++

```

Run generate_density.py in data_preprocess to generate ground-truth density map


Please put the image and ground-truth in the same folder
```python

Data_root/
         -train/
               -IMG_1.h5
               -IMG_1.jpg
               ⋮
         -test/
               -IMG_1.h5
               -IMG_1.jpg
               ⋮
 ⋮

```

Run the data_pair.py to generate data_list
```python

python data_preprocess/data_pair.py

```

## Pretrained model on ShanghaiTech Part A can downloade at here
["Here"](https://drive.google.com/drive/folders/1URV04UehpIASURLM8V89DVGrOncy3Lei)

## Backbone pretrained model
["Here"](https://drive.google.com/drive/u/4/folders/1QeLZc7_4TZVZ7awRQXNmgvGtPl6OGUAR)

## Training
```python

python train.py --data_root 'data_root' --epochs 4000

```

## Run testing
```python

python test.py --weight_path 'checkpoint_path'

```

## Quantitative comparison


<img src="" width="1337" height="449">

Quantitative evaluation on four dataset. We report Mean Abosolute Error (MAE), Root Mean Square Error (RMSE). (Bold means the 1st best; Underline means the 2nd best).


## Qualitative comparisons

- Visualize

<img src="" width="1279" height="400">

The generated density map comparison of our method and some other methods on ShanghaiTech PartA dataset. From left to right are input image, ground truth, MCNN, CSRnet, CAN, BL, DM-count, and Ours.


## Ablation study

- Ablation study 

<div align=center>
<img src="" width="546" height="195">
</div>

Ablation study of all modual we used with size 128x128 images on ShanghaiTech PartA dataset. We report Mean Abosolute Error (MAE), Root Mean Square Error (RMSE). (Bold means the 1st best; Underline means the 2nd best)
