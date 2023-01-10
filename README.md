# DataIsPower

## 介绍

## 环境准备

```bash
conda create --name DataIsPower python=3.8
conda activate DataIsPower
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
```

## 训练

首先运行run_deepbbt.sh生成soft prompt文件

```bash
nohup bash run_deepbbt_<task_name>.sh > run_deepbbt_<task_name>.log 2>&1 &
```

## 测试

运行完毕后，运行test_deepbbt.sh生成最后的prediction文件

```bash
nohup bash test_deepbbt_<task_name>.sh > test_deepbbt_<task_name>.log 2>&1 &
```

### 注意

- 调incontext的话data要重新加载不能从caches里拿，所以——refresh==True
