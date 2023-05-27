# Make Prompt-based Black-Box Tuning Colorful: Boosting Model Generalization from Three Orthogonal Perspectives

Work in Progress

## Updates

- 2023/05/03: Release the first version of BBT-RGB, please check our [paper](qiushisun.github.io). 🌈



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



## Acknowledgement

Part of the codes are adapted from [Black-Box-Tuning](https://github.com/txsun1997/Black-Box-Tuning), and we would like to extend our sincere appreciation to Xiang Li and Ming Gao for their invaluable suggestions regarding the methodology and presentation of this paper.



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

