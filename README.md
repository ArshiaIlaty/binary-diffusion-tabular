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

## Pip install from repository

```bash
pip install git+https://github.com/vkinakh/binary-diffusion-tabular.git
````

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
# Project Tree
```
binary-diffusion-fork
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ branches
│  ├─ config
│  ├─ description
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     └─ remotes
│  │        └─ origin
│  │           ├─ HEAD
│  │           └─ main
│  ├─ objects
│  │  ├─ 0c
│  │  │  └─ c835ce3a73e8cc593e8be92c315875449cad3b
│  │  ├─ 0d
│  │  │  └─ 025cfbad30820e68efb04f4b339262e5e669fa
│  │  ├─ 0f
│  │  │  └─ 287626535fb343ee1f6e3f147ea6677584c23d
│  │  ├─ 16
│  │  │  └─ aec9cdb8ae807b4cd0a3c2fbbf7d21924f0aba
│  │  ├─ 2a
│  │  │  └─ cb4b401318b37586bd34429e643b107c6fcd73
│  │  ├─ 2b
│  │  │  └─ 553aaeb502e7b8677b3ffb8e03481d9ec165e3
│  │  ├─ 3a
│  │  │  └─ 6e2b2d2af19c6b9e2bfaa327a49138973f5e71
│  │  ├─ 3c
│  │  │  ├─ 43c4541009e454a1e7e2c13964a88f637ee19a
│  │  │  └─ cc4c6554443b403e57a75dd238cc1ffcb0b454
│  │  ├─ 46
│  │  │  └─ 8909b87592940f64391506e73192f8722455f0
│  │  ├─ 51
│  │  │  └─ 8e169a74870093cbd82a1fc3b3c43b29754d64
│  │  ├─ 57
│  │  │  └─ 6f9c078a13f9ce28ffb51b4cb0308d9084d7f8
│  │  ├─ 68
│  │  │  └─ eae44c7af1eb3d43140c755f460010ba23e763
│  │  ├─ 6b
│  │  │  └─ e74c032dc101b633d88f966cddc9b8a7c7d053
│  │  ├─ 73
│  │  │  └─ bec7ca591752aebb5e940fb886d78525d823e7
│  │  ├─ 7a
│  │  │  └─ ff6b7ddb9e40317609008ff9bb3ab0440632a7
│  │  ├─ 94
│  │  │  └─ f51d54ca1e26532169ff1998666921058f0a42
│  │  ├─ 98
│  │  │  └─ 37e7f8c71d0a6c72d27c65cfd2865d4bc32067
│  │  ├─ a3
│  │  │  └─ cb6ea4798db486886d1cb2fc9976c397f3fc5b
│  │  ├─ b4
│  │  │  └─ b3cffb87c8ba5bea496858f8966d03127120a3
│  │  ├─ bf
│  │  │  └─ 00c5b45d4d9d7acbe86e99147ac005a64f34d8
│  │  ├─ c1
│  │  │  └─ 1c0569c98c6d54aa060898fb9c405d3863a705
│  │  ├─ c8
│  │  │  └─ beae82126f1f178b015ed2c313d94825d194de
│  │  ├─ d1
│  │  │  └─ 6e70c0946bd56e3a01957262fd97e6e4001ce7
│  │  ├─ d7
│  │  │  └─ 5a5e4cdfa826abbb9ff30b1e80a880cb082503
│  │  ├─ e3
│  │  │  └─ 053567e401fad82c413958c606795b711629d2
│  │  ├─ e5
│  │  │  └─ 6abb6303084744726ba895652e3b70c5c71a71
│  │  ├─ ea
│  │  │  └─ f17a5f8d5df05ce69467ce86119f968d096ba7
│  │  ├─ f4
│  │  │  └─ 8969f93da7ac152b105843e13e058fc4b4321e
│  │  ├─ fa
│  │  │  └─ 37bd46659b9fbb282fbaacc8c1af50fa0d6fa9
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-5d67c6c08c88b5504fd4aaffbfa0de24d18db20e.idx
│  │     └─ pack-5d67c6c08c88b5504fd4aaffbfa0de24d18db20e.pack
│  ├─ packed-refs
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ HEAD
│     │     └─ main
│     └─ tags
├─ .gitignore
├─ LICENSE
├─ README.md
├─ binary_diffusion_tabular
│  ├─ __init__.py
│  ├─ dataset.py
│  ├─ diffusion.py
│  ├─ model.py
│  ├─ trainer.py
│  ├─ transformation.py
│  └─ utils.py
├─ checkNan.py
├─ command_history.txt
├─ configs
│  ├─ adult.yaml
│  ├─ cervical.yaml
│  ├─ diabetes.yaml
│  ├─ heloc.yaml
│  ├─ housing.yaml
│  ├─ sick.yaml
│  └─ travel.yaml
├─ data
│  ├─ adult.csv
│  ├─ adult_test.csv
│  ├─ adult_train.csv
│  ├─ cervical.csv
│  ├─ cervical_test.csv
│  ├─ cervical_train.csv
│  ├─ diabetes.csv
│  ├─ diabetes_test.csv
│  ├─ diabetes_train.csv
│  ├─ heloc.csv
│  ├─ heloc_test.csv
│  ├─ heloc_train.csv
│  ├─ housing.csv
│  ├─ housing_test.csv
│  ├─ housing_train.csv
│  ├─ sick.csv
│  ├─ sick_test.csv
│  ├─ sick_train.csv
│  ├─ travel.csv
│  ├─ travel_test.csv
│  └─ travel_train.csv
├─ divisionbyZeroError.py
├─ environment.yml
├─ federated_binary_diffusion
│  ├─ configs
│  │  ├─ client1_config.yaml
│  │  └─ client2_config.yaml
│  ├─ data
│  │  ├─ client1_data.csv
│  │  └─ client2_data.csv
│  ├─ data_prep.py
│  ├─ evaluate_synthetic.py
│  ├─ federated_training.py
│  ├─ federated_training_scratch.py
│  ├─ generate_synthetic_data.py
│  ├─ models
│  │  ├─ client1
│  │  ├─ client2
│  │  └─ global
│  │     └─ global.py
│  └─ utils.py
├─ pyproject.toml
├─ requirements.txt
├─ sample.py
├─ tests
│  ├─ test_fixed_size_binary_table_transformation.py
│  └─ test_model.py
└─ train.py

```