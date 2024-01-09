All the code and data will be released in this week 🤗. (before 2024.1.14) 

## Towards Effective Multiple-in-One Image Restoration: A Sequential and Prompt Learning Strategy

<a href='https://arxiv.org/abs/2401.03379'><img src='https://img.shields.io/badge/arXiv-2401.03379-b31b1b.svg'></a> &nbsp;&nbsp;

Authors: Xiangtao Kong, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN) and [Lei Zhang](https://scholar.google.com/citations?user=tAK5l1IAAAAJ&hl=en&oi=ao)

## Abstract
While single task image restoration (IR) has achieved significant successes, it remains a challenging issue to train a single model which can tackle multiple IR tasks. In this work, we investigate in-depth the multiple-in-one (MiO) IR problem, which comprises seven popular IR tasks. We point out that MiO IR faces two pivotal challenges: the optimization of diverse objectives and the adaptation to multiple tasks. To tackle these challenges, we present two simple yet effective strategies. The first strategy, referred to as sequential learning, attempts to address how to optimize the diverse objectives, which guides the network to incrementally learn individual IR tasks in a sequential manner rather than mixing them together. The second strategy, i.e., prompt learning, attempts to address how to adapt to the different IR tasks, which assists the network to understand the specific task and improves the generalization ability. By evaluating on 19 test sets, we demonstrate that the sequential and prompt learning strategies can significantly enhance the MiO performance of commonly used CNN and Transformer backbones. Our experiments also reveal that the two strategies can supplement each other to learn better degradation representations and enhance the model robustness. It is expected that our proposed MiO IR formulation and strategies could facilitate the research on how to train IR models with higher generalization capabilities.

## 🔎 Overview framework
![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/MiOIR.png)

## 📌 Quantitative Results

![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/performance.png)

More results of Restormer, Uformer and PromptIR can be found in paper.

## 📷 Visual Results
![Demo Image](https://github.com/Xiangtaokong/MiOIR/blob/main/demo_images/visual_00.png)



## ❤️ Acknowledgments
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR).

## 📧 Contact
If you have any questions, please feel free to contact: `xiangtao.kong@connect.polyu.hk`


## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).


<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Xiangtaokong/MiOIR)

</details>


