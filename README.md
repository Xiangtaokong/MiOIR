# MiOIR
Towards Effective Multiple-in-One Image Restoration: A Sequential and Prompt Learning Strategy.

[Paper](...)

Authors: Xiangtao Kong, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN), [Lei Zhang](http://www4.comp.polyu.edu.hk/~cslzhang/)


![Demo Image](https://raw.githubusercontent.com/Xiangtaokong/ClassSR/main/demo_images/show.png)

## Abstract
While single task image restoration (IR) has achieved significant successes, it remains a challenging issue to train a single model which can tackle multiple IR tasks. In this work, we investigate in-depth the multiple-in-one (MiO) IR problem, which comprises seven popular IR tasks, and present two simple yet effective strategies to enhance the network learning performance. The first strategy, referred to as sequential learning, guides the network to incrementally learn individual IR tasks in a sequential manner rather than mixing them together. The second strategy, i.e., prompt learning, assists the network to understand the specific task for the image being processed, and hence improves the generalization capability of trained IR models. We demonstrate that the sequential and prompt learning strategies can significantly enhance the MiO performance of commonly used CNN and Transformer backbones, such as SRResNet and SwinIR, with improvements up to 1.21 dB/1.07 dB on in/out-of-distribution test sets, respectively. They can also enhance the state-of-the-art method PromptIR by 1.1 dB with only 75\% of its parameters. Our experiments also reveal that sequential and prompt learning can supplement each other to learn degradation representations and enhance the model robustness. It is expected that our proposed strategies and findings could facilitate the research on how to train IR models with higher generalization capabilities.

## Dependencies

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

# Codes 
- Our codes version based on [BasicSR](https://github.com/xinntao/BasicSR). 

## How to test a single branch
1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```
2. Download the testing datasets ([DIV2K_valid](https://data.vision.ee.ethz.ch/cvl/DIV2K/)). 

3. Download the [divide_val.log](https://drive.google.com/file/d/1zMDD9Z_-fM2R2qm2QLoq7N2LMG6V92JT/view?usp=sharing) and move it to `.codes/data_scripts/`.

4. Generate simple, medium, hard (class1, class2, class3) validation data. 
```
cd codes/data_scripts
python extract_subimages_test.py
python divide_subimages_test.py
```
5. Download [pretrained models](https://drive.google.com/drive/folders/1jzAFazbaGxHb-xL4vmxc-hHbR1J-uek_?usp=sharing) and move them to  `./experiments/pretrained_models/` folder. 

6. Run testing for a single branch.
```
cd codes
python test.py -opt options/test/test_FSRCNN.yml
python test.py -opt options/test/test_CARN.yml
python test.py -opt options/test/test_SRResNet.yml
python test.py -opt options/test/test_RCAN.yml
```

7. The output results will be sorted in `./results`. 

## How to test ClassSR
1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```

2. Download the testing datasets (Test2K, 4K, 8K) [Google Drive](https://drive.google.com/drive/folders/18b3QKaDJdrd9y0KwtrWU2Vp9nHxvfTZH?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1OARDfd2x3ynQs7m1tu_RnA) (Password: 7dw1) .

3. You can also download the source data [DIV8K](https://competitions.codalab.org/competitions/22217#participate). Test8K contains the images (index 1401-1500) from DIV8K. Test2K/4K contain the images (index 1201-1300/1301-1400) from DIV8K which are downsampled to 2K and 4K resolution. (In this way, you need register for the competition (Ntire 2020 was held on 2020, but we can register now), then you can download DIV8K dataset.)

4. Download [pretrained models](https://drive.google.com/drive/folders/1jzAFazbaGxHb-xL4vmxc-hHbR1J-uek_?usp=sharing) and move them to  `./experiments/pretrained_models/` folder. 

5. Run testing for ClassSR.
```
cd codes
python test_ClassSR.py -opt options/test/test_ClassSR_FSRCNN.yml
python test_ClassSR.py -opt options/test/test_ClassSR_CARN.yml
python test_ClassSR.py -opt options/test/test_ClassSR_SRResNet.yml
python test_ClassSR.py -opt options/test/test_ClassSR_RCAN.yml
```
6. The output results will be sorted in `./results`. 


## How to train a single branch
1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```
2. Download the training datasets([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) and validation dataset(Set5).

3. Download the [divide_train.log](https://drive.google.com/file/d/1WhyYYZHfpoNEjslojuqZLR46Nlr15zqQ/view?usp=sharing) and move it to `.codes/data_scripts/`.

4. Generate simple, medium, hard (class1, class2, class3) training data. 
```
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
python divide_subimages_train.py
```

5. Run training for a single branch (default branch1, the simplest branch).
```
cd codes
python train.py -opt options/train/train_FSRCNN.yml
python train.py -opt options/train/train_CARN.yml
python train.py -opt options/train/train_SRResNet.yml
python train.py -opt options/train/train_RCAN.yml
```
6. The experiments will be sorted in `./experiments`. 

## How to train ClassSR

1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```
2. Download the training datasets ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) and validation dataset([DIV2K_valid](https://data.vision.ee.ethz.ch/cvl/DIV2K/), index 801-810). 


3. Generate training data (the all data(1.59M) in paper).
```
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
```
4. Download [pretrained models](https://drive.google.com/drive/folders/1jzAFazbaGxHb-xL4vmxc-hHbR1J-uek_?usp=sharing)(pretrained branches) and move them to  `./experiments/pretrained_models/` folder. 

5. Run training for ClassSR.
```
cd codes
python train_ClassSR.py -opt options/train/train_ClassSR_FSRCNN.yml
python train_ClassSR.py -opt options/train/train_ClassSR_CARN.yml
python train_ClassSR.py -opt options/train/train_ClassSR_SRResNet.yml
python train_ClassSR.py -opt options/train/train_ClassSR_RCAN.yml
```
6. The experiments will be sorted in `./experiments`. 

## How to generate demo images

Generate demo images like this one:

![Demo Image](https://raw.githubusercontent.com/Xiangtaokong/ClassSR/main/demo_images/show.png)

Change the 'add_mask: False' to True in test_ClassSR_xxx.yml and run testing for ClassSR.

## Citation
```
XXXX
```

## Contact
Email: xiangtao.kong@connect.polyu.hk

