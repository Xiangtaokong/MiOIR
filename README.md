
## Towards Effective Multiple-in-One Image Restoration: A Sequential and Prompt Learning Strategy

<a href='https://arxiv.org/abs/2401.03379'><img src='https://img.shields.io/badge/arXiv-2401.03379-b31b1b.svg'></a> &nbsp;&nbsp;

Authors: Xiangtao Kong, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN) and [Lei Zhang](https://scholar.google.com/citations?user=tAK5l1IAAAAJ&hl=en&oi=ao)

## Abstract
While single task image restoration (IR) has achieved significant successes, it remains a challenging issue to train a single model which can tackle multiple IR tasks. In this work, we investigate in-depth the multiple-in-one (MiO) IR problem, which comprises seven popular IR tasks. We point out that MiO IR faces two pivotal challenges: the optimization of diverse objectives and the adaptation to multiple tasks. To tackle these challenges, we present two simple yet effective strategies. The first strategy, referred to as sequential learning, attempts to address how to optimize the diverse objectives, which guides the network to incrementally learn individual IR tasks in a sequential manner rather than mixing them together. The second strategy, i.e., prompt learning, attempts to address how to adapt to the different IR tasks, which assists the network to understand the specific task and improves the generalization ability. By evaluating on 19 test sets, we demonstrate that the sequential and prompt learning strategies can significantly enhance the MiO performance of commonly used CNN and Transformer backbones. Our experiments also reveal that the two strategies can supplement each other to learn better degradation representations and enhance the model robustness. It is expected that our proposed MiO IR formulation and strategies could facilitate the research on how to train IR models with higher generalization capabilities.

:star: If SeeSR is helpful to your images or projects, please help star this repo. Thanks! :hugs:

## 🔎 Overview framework
![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/MiOIR.png)

## 📌 Quantitative Results

![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/performance.png)

More results of `Restormer`, `Uformer` and `PromptIR` can be found in paper.

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
conda install -r requirements.txt (or refer to the environment of [BasicSR](https://github.com/XPixelGroup/BasicSR))
```

## 🚀 Test

#### Setp 1 Download the pre-trained models

[One Drive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/23105237r_connect_polyu_hk/EmwjMLnXyztLo8dmBcYA6xsBikhDiVPM3Oc8cyrb2EWxzA?e=05xij8)

[Baidu Drive](https://pan.baidu.com/s/1OCtPAv8sZe27mxBs-5HT_w?pwd=yxuw).    Key: yxuw

We provide 27 pre-trained models (including `SRResNet`, `SwinIR`, `Restormer`, `Uformer` and `PromptIR`) that appear in the paper.

#### Setp 2 Download the testsets

[One Drive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/23105237r_connect_polyu_hk/Es-ie8Hd_O5KuLDXH1G7c-4BHok3MH8A43-NltzBblb83A?e=dZT3RU)

[Baidu Drive](https://pan.baidu.com/s/1OCtPAv8sZe27mxBs-5HT_w?pwd=yxuw).    Key: yxuw

Please download `MiO_test.zip` and unzip it to `MiOIR/data`.

#### Setp 3 Edit the test yml file

Edit the yml files in `MiOIR/options/test` that you need. 

Please modify the path of your model and data.

#### Setp 4 Run the command

```
cd MiOIR
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py --launcher none -opt MiOIR/options/test/xxxx.yml
```
The results will be put in `MiO/results`.

## :star: Train 

#### Step1: Download the training data

[One Drive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/23105237r_connect_polyu_hk/Eiic-eLBEXNEiB_-UPVGDNcBJ5jHiMM5V4oYAL7u1KFxRg?e=eVcsAP)

[Baidu Drive](https://pan.baidu.com/s/1OCtPAv8sZe27mxBs-5HT_w?pwd=yxuw).    Key: yxuw

Download all the data (including `sr, blur, noise, jpeg, rain, haze, dark, GT .zip`), intotal 120G for dowmload.

OR only download `GT.zip` and `depth.zip`, then generate the rest data by `MiOIR/data_script/add_MiO_train_degradation.py`, intotal 16G for dowmload.

#### Step2: Data prepare

Use `data_script/gen_sub.py` to crop the data to `300x300`, modify some path and make sure save them with floder name of `xxx_sub300`.

Use `data_script/gen_meta.py` to generate training meta file, modify some path and the order of the sequential learning (`_S`) you want.

#### Step3: Edit the train yml file

Edit the yml files in `MiOIR/options/train` that you need.Please modify the path of your model and data.

Note that: 

Because the sequential learning (`_S`) need a series of yml files, please edit mixed learning (`_M`) first and then use `options/gen_train.py` to generate the yml files.

#### Step4: Run the command

```
cd MiOIR
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt xxxxxx.yml
```
Note to modify the GPU card index and number `--nproc_per_node= GPU number` (correspond to it in yml).

If you use `options/gen_train.py` to generate yml files, it will also generate a `.sh` script in `MiOIR/run_sh`. 

You could train Sequential Learning models through the script conveniently.

## ❤️ Acknowledgments
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR).

## 📧 Contact
If you have any questions, please feel free to contact: `xiangtao.kong@connect.polyu.hk`

## 🎓Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```
@article{kong2024towards,
  title={Towards Effective Multiple-in-One Image Restoration: A Sequential and Prompt Learning Strategy},
  author={Kong, Xiangtao and Dong, Chao and Zhang, Lei},
  journal={arXiv preprint arXiv:2401.03379},
  year={2024}
}
```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).




<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Xiangtaokong/MiOIR)

</details>


