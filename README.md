# Tabular Data Generation using Binary Diffusion

This repository contains the official implementation of the paper "Tabular Data Generation using Binary Diffusion", 
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

COMING SOON

# Quickstart

COMING SOON

# Citation
```
@article{kinakh2024tabular,
  title={Tabular Data Generation using Binary Diffusion},
  author={Kinakh, Vitaliy and Voloshynovskiy, Slava},
  journal={arXiv preprint arXiv:2409.13882},
  year={2024}
}
```