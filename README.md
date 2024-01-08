
## Towards Effective Multiple-in-One Image Restoration: A Sequential and Prompt Learning Strategy

<a href=''><img src='https://img.shields.io/badge/arXiv-2311.16518-b31b1b.svg'></a> &nbsp;&nbsp;

Authors: Xiangtao Kong, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN) and [Lei Zhang](https://scholar.google.com/citations?user=tAK5l1IAAAAJ&hl=en&oi=ao)

## Abstract
While single task image restoration (IR) has achieved significant successes, it remains a challenging issue to train a single model which can tackle multiple IR tasks. In this work, we investigate in-depth the multiple-in-one (MiO) IR problem, which comprises seven popular IR tasks. We point out that MiO IR faces two pivotal challenges: the optimization of diverse objectives and the adaptation to multiple tasks. To tackle these challenges, we present two simple yet effective strategies. The first strategy, referred to as sequential learning, attempts to address how to optimize the diverse objectives, which guides the network to incrementally learn individual IR tasks in a sequential manner rather than mixing them together. The second strategy, i.e., prompt learning, attempts to address how to adapt to the different IR tasks, which assists the network to understand the specific task and improves the generalization ability. By evaluating on 19 test sets, we demonstrate that the sequential and prompt learning strategies can significantly enhance the MiO performance of commonly used CNN and Transformer backbones. Our experiments also reveal that the two strategies can supplement each other to learn better degradation representations and enhance the model robustness. It is expected that our proposed MiO IR formulation and strategies could facilitate the research on how to train IR models with higher generalization capabilities.

## 🔎 Overview framework
![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/MiOIR.png)

## 📌 Quantitative Results
![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/performance.png)

## 📷 Visual Results
![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/visual_00.png)



## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/Xiangtaokong/MiOIR.git
cd MiOIR

# create an environment with python >= 3.8
conda create -n MiOIR python=3.8
conda activate MiOIR
pip install -r requirements.txt
```

## 🚀 Test

#### Setp 1 Download the pretrained models

Google Drive: coming soon.
Baidu Drive: [link]()

#### Setp 2 Download the testsets

Google Drive: coming soon.
Baidu Drive: [link]()

#### Setp 3 Edit the test ymal file

Edit the 

#### Setp 4 Run the command

## :star: Train 

#### Step1: Download the training data
Download the pretrained [SD-2.1base models](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth). You can put them into `preset/models`.

#### Step2: Edit the train ymal file
We pre-prepare training data pairs for the training process, which would take up some memory space but save training time. We train the DAPE with [COCO](https://cocodataset.org/#home) and train the SeeSR with common low-level datasets, such as DF2K.



#### Step3: Run the command
Please specify the DAPE training data path at `line 13` of `basicsr/options/dape.yaml`, then run the training command:
```
python basicsr/train.py -opt basicsr/options/dape.yaml
```
You can modify the parameters in `dape.yaml` to adapt to your specific situation, such as the number of GPUs, batch size, optimizer selection, etc. For more details, please refer to the settings in Basicsr. 

#### Step4: Training for SeeSR



## ❤️ Acknowledgments
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR).

## 📧 Contact
If you have any questions, please feel free to contact: `xiangtao.kong@connect.polyu.hk`

## 🎓Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```

```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).


<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=cswry/SeeSR)

</details>


XXXX
```

## Contact
Email: xiangtao.kong@connect.polyu.hk

