# Make Prompt-based Black-Box Tuning Colorful: Boosting Model Generalization from Three Orthogonal Perspectives

[![arXiv](https://img.shields.io/badge/arXiv-2305.08088-b31b1b.svg)](https://arxiv.org/abs/2305.08088) 
![License](https://img.shields.io/badge/License-MIT-blue)

## Updates

- 2024/01/20: BBT-RGB is accepted by LREC-COLING 2024. 🎉
- 2023/05/03: Release the first version of BBT-RGB, please check our [paper](https://arxiv.org/abs/2305.08088). 🌈



## Introduction

We describe BBT-RGB in this paper, a suite of straightforward and complementary techniques for enhancing the efficiency and performance of black-box optimization. Specifically, our method includes three plug-and-play components: (1) Two-stage derivative-free optimization strategy that facilitates fast convergence and mitigates overfitting; (2) Automatic verbalizer construction with its novel usage under few-shot settings; (3) Better prompt initialization policy based on instruction search and auto-selected demonstration.



<img src="./images/BBT-RGB-Overview.png" alt="BBT-RGB-Overview" style="zoom:20%;" />



## Preparing the Environment

```bash
conda create --name bbtrgb python=3.8
conda activate bbtrgb
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
```


## Performance

Them main results on RoBERTa-Large are shown below. The best results are in bold. Some baselines are collected from [Black-Box-Tuning](https://github.com/txsun1997/Black-Box-Tuning).

| Method            | Tunable Params | SST-2 acc  | Yelp P. acc | AG's News acc | DBPedia acc | MRPC F1 | SNLI acc | RTE acc | Avg.   |
|-------------------|----------------|------------|-------------|---------------|-------------|---------|----------|---------|--------|
| Model Tuning      | 355M           | 85.39±2.84  | 91.82±0.79   | 86.36±1.85     | 97.98±0.14   | 77.35±5.70 | 54.64±5.29 | 58.60±6.21 | **78.88** |
| Prompt Tuning     | 50K            | 68.23±3.78  | 61.02±6.65   | 84.81±0.66     | 87.75±1.48   | 51.61±8.67 | 36.13±1.51 | 54.69±3.79 | 63.46  |
| P-Tuning v2       | 1.2M           | 64.33±3.05  | **92.63±1.39** | 83.46±1.01     | 97.05±0.41   | 68.14±3.89 | 36.89±0.79 | 50.78±2.28 | 70.47  |
| Adapter           | 2.4M           | 83.91±2.90  | 90.99±2.86   | 86.01±2.18     | **97.99±0.07** | 69.20±3.58 | 57.46±6.63 | 48.62±4.74 | 76.31  |
| LoRA              | 786K           | **88.49±2.90** | 90.21±4.00   | **87.09±0.85** | 97.86±0.17   | 72.14±2.23 | **61.03±8.55** | 49.22±5.12 | **78.01** |
| BitFit            | 172K           | 81.19±6.08  | 88.63±6.69   | 86.83±0.62     | 94.42±0.94   | 66.26±6.81 | 53.42±10.63| 52.59±5.31 | 74.76  |
| Manual Prompt     | 0              | 79.82       | 89.65        | 76.96          | 41.33        | 67.40    | 31.11     | 51.62    | 62.56  |
| In-Context Learning | 0            | 79.79±3.06  | 85.38±3.92   | 62.21±13.46    | 34.83±7.59   | 45.81±6.67 | 47.11±0.63 | 60.36±1.56 | 59.36  |
| BBT               | 500            | 89.56±0.25  | 91.50±0.16   | 81.51±0.79     | 79.99±2.95   | 61.56±4.34 | 46.58±1.33 | 52.59±2.21 | 71.90  |
| BBTv2             | 12K            | 90.33±1.73  | 92.86±0.62   | 85.28±0.49     | 93.64±0.68   | 77.01±4.73 | 57.27±2.27 | 56.68±3.32 | 79.01  |
| **BBT-RGB**| 12K            | **92.89±0.26** | **94.20±0.48** | **85.60±0.41** | **94.41±0.73** | **79.49±1.84** | **60.71±0.66** | **61.82±1.20** | **81.30** |



## Acknowledgement

This is also derived from a prize-winning solution of the [First International Algorithm Case Competition: PLM Tuning Track, Guangdong-Hong Kong-Macao Greater Bay Area](https://iacc.pazhoulab-huangpu.com/). Part of the codes are adapted from [Black-Box-Tuning](https://github.com/txsun1997/Black-Box-Tuning).

## Citation

Please consider citing us if you find this repository useful.👇

```bibtex
@misc{sun2023bbtrgb,
      title         = {Make Prompt-based Black-Box Tuning Colorful: Boosting Model Generalization from Three Orthogonal Perspectives}, 
      author        = {Qiushi Sun and Chengcheng Han and Nuo Chen and Renyu Zhu and Jingyang Gong and Xiang Li and Ming Gao},
      year          = {2023},
      eprint        = {2305.08088},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CL}
}
```

