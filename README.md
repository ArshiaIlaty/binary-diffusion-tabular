[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tabular-data-generation-using-binary/tabular-data-generation-on-adult-census)](https://paperswithcode.com/sota/tabular-data-generation-on-adult-census?p=tabular-data-generation-using-binary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tabular-data-generation-using-binary/tabular-data-generation-on-diabetes)](https://paperswithcode.com/sota/tabular-data-generation-on-diabetes?p=tabular-data-generation-using-binary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tabular-data-generation-using-binary/tabular-data-generation-on-travel)](https://paperswithcode.com/sota/tabular-data-generation-on-travel?p=tabular-data-generation-using-binary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tabular-data-generation-using-binary/tabular-data-generation-on-sick)](https://paperswithcode.com/sota/tabular-data-generation-on-sick?p=tabular-data-generation-using-binary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tabular-data-generation-using-binary/tabular-data-generation-on-california-housing)](https://paperswithcode.com/sota/tabular-data-generation-on-california-housing?p=tabular-data-generation-using-binary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tabular-data-generation-using-binary/tabular-data-generation-on-heloc)](https://paperswithcode.com/sota/tabular-data-generation-on-heloc?p=tabular-data-generation-using-binary)

# Tabular Data Generation using Binary Diffusion

This repository contains the official implementation of the paper "[Tabular Data Generation using Binary Diffusion](https://arxiv.org/abs/2409.13882)", 
accepted to [3rd Table Representation Learning Workshop @ NeurIPS 2024](https://table-representation-learning.github.io/).

# Abstract

Generating synthetic tabular data is critical in machine learning, especially when real data is limited or sensitive. 
Traditional generative models often face challenges due to the unique characteristics of tabular data, such as mixed 
data types and varied distributions, and require complex preprocessing or large pretrained models. In this paper, we 
introduce a novel, lossless binary transformation method that converts any tabular data into fixed-size binary 
representations, and a corresponding new generative model called Binary Diffusion, specifically designed for binary 
data. Binary Diffusion leverages the simplicity of XOR operations for noise addition and removal and employs binary 
cross-entropy loss for training. Our approach eliminates the need for extensive preprocessing, complex noise parameter 
tuning, and pretraining on large datasets. We evaluate our model on several popular tabular benchmark datasets, 
demonstrating that Binary Diffusion outperforms existing state-of-the-art models on Travel, Adult Income, and Diabetes 
datasets while being significantly smaller in size.

# Installation

## Local conda installation

Conda environment was tested on Ubuntu 22.04 LTS and on Mac OS Sonoma 14.6 with M3 chip.

```bash
conda env create -f environment.yml
```

## Local pip installation

```bash
pip install -r requirements.txt
```

# Quickstart

## Run tests
```bash
python -m unittest discover -s ./tests
```

## Training

To use [train.py](train.py) script, first fill configuration fill. See examples in [configs](configs). Then run:
```bash
python train.py -c=<path to config>
```

## Example training script
```python
import pandas as pd
import wandb

from binary_diffusion_tabular import (
    FixedSizeBinaryTableDataset, 
    SimpleTableGenerator, 
    BinaryDiffusion1D, 
    FixedSizeTableBinaryDiffusionTrainer,
    drop_fill_na
)

df = pd.read_csv("./data/adult_train.csv")
columns_numerical = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

columns_categorical = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

task = "classification"
column_target = "label"

df = drop_fill_na(
    df=df,
    columns_numerical=columns_numerical,
    columns_categorical=columns_categorical,
    dropna=True,
    fillna=False
)

dataset = FixedSizeBinaryTableDataset(
    table=df,
    target_column=column_target,    # conditional generation
    split_feature_target=True,
    task=task,
    numerical_columns=columns_numerical,
    categorical_columns=columns_categorical
)

classifier_free_guidance = True
target_diffusion = "two_way"

dim = 256
n_res_blocks = 3
device = "cuda"

model = SimpleTableGenerator(
    data_dim=dataset.row_size,
    dim=dim,
    n_res_blocks=n_res_blocks,
    out_dim=(
        dataset.row_size * 2
        if target_diffusion == "two_way"
        else dataset.row_size
    ),
    task=task,
    conditional=dataset.conditional,
    n_classes=0 if task == "regression" else dataset.n_classes,
    classifier_free_guidance=classifier_free_guidance,
).to(device)

schedule = "quad"
n_timesteps = 1000

diffusion = BinaryDiffusion1D(
    denoise_model=model,
    schedule="quad",
    n_timesteps=n_timesteps,
    target=target_diffusion
).to(device)

logger = wandb.init(
    project="your-project",
    config={"key": "value"},
    name="adult_CFG"
)

num_training_steps = 500_000
log_every = 100
save_every = 10_000
save_num_samples = 64
ema_decay = 0.995
ema_update_every = 10
lr = 1e-4
opt_type = "adam"
batch_size = 256
n_workers = 16
zero_token_probability = 0.1
results_folder = "./results/adult_CFG"

trainer = FixedSizeTableBinaryDiffusionTrainer(
    diffusion=diffusion,
    dataset=dataset,
    train_num_steps=num_training_steps,
    log_every=log_every,
    save_every=save_every,
    save_num_samples=save_num_samples,
    max_grad_norm=None,
    gradient_accumulate_every=1,
    ema_decay=ema_decay,
    ema_update_every=ema_update_every,
    lr=lr,
    opt_type=opt_type,
    batch_size=batch_size,
    dataloader_workers=n_workers,
    classifier_free_guidance=classifier_free_guidance,
    zero_token_probability=zero_token_probability,
    logger=logger,
    results_folder=results_folder
)

trainer.train()
```

## Sampling

To use [sample.py](sample.py) script, you need a pretrained model and data transformation. Then run
```bash
python sample.py \
       --ckpt=<path to model ckpt> \
       --ckpt_transformation=<path to transformation ckpt> \
       --n_timesteps=<number of sampling steps> \
       --out=<path to output folder> \
       --n_samples=<number of rows to generate> \
       --batch_size=<sampling batch size> \
       --threshold=<threshold for binaryzation of logits> \     # 0.5 default
       --strategy=<sampling strategy> \                         # target or mask
       --seed=<seed for reproducibility> \                      # default no seed
       --guidance_scale=<scale for classifier free guidance> \  # 0 default, no classifier free guidance
       --target_column_name=<name of target column> \           # name of target column, in case of conditional generation
       --device=<device to run on> \
       --use_ema                                                # whether to use EMA diffusion model
```

# Results

# Results

The table below presents the **Binary Diffusion** results across various datasets and models. Performance metrics are shown as **mean ± standard deviation**.

| **Dataset**             | **LR (Binary Diffusion)** | **DT (Binary Diffusion)** | **RF (Binary Diffusion)** | **Params** | **Model link**                                                                      |
|-------------------------|---------------------------|---------------------------|---------------------------|------------|-------------------------------------------------------------------------------------|
| **Travel**              | **83.79 ± 0.08**          | **88.90 ± 0.57**          | **89.95 ± 0.44**          | **1.1M**   | [Link](https://huggingface.co/vitaliykinakh/binary-ddpm-tabular/tree/main/travel)   |
| **Sick**                | 96.14 ± 0.63              | **97.07 ± 0.24**          | 96.59 ± 0.55              | **1.4M**   | [Link](https://huggingface.co/vitaliykinakh/binary-ddpm-tabular/tree/main/sick)     |
| **HELOC**               | 71.76 ± 0.30              | 70.25 ± 0.43              | 70.47 ± 0.32              | **2.6M**   | [Link](https://huggingface.co/vitaliykinakh/binary-ddpm-tabular/tree/main/heloc)    |
| **Adult Income**        | **85.45 ± 0.11**          | **85.27 ± 0.11**          | **85.74 ± 0.11**          | **1.4M**   | [Link](https://huggingface.co/vitaliykinakh/binary-ddpm-tabular/tree/main/adult)    |
| **Diabetes**            | **57.75 ± 0.04**          | **57.13 ± 0.15**          | 57.52 ± 0.12              | **1.8M**   | [Link](https://huggingface.co/vitaliykinakh/binary-ddpm-tabular/tree/main/diabetes) |
| **California Housing**  | *0.55 ± 0.00*             | 0.45 ± 0.00               | 0.39 ± 0.00               | **1.5M**   | [Link](https://huggingface.co/vitaliykinakh/binary-ddpm-tabular/tree/main/housing)  |

---

# Citation
```
@article{kinakh2024tabular,
  title={Tabular Data Generation using Binary Diffusion},
  author={Kinakh, Vitaliy and Voloshynovskiy, Slava},
  journal={arXiv preprint arXiv:2409.13882},
  year={2024}
}
```