# CNN vs Vision Transformer Study

This project compares **Convolutional Neural Networks (ResNet-18)** and **Vision Transformers (ViT-B/16)** across several vision datasets.

The notebook implements a unified PyTorch training pipeline and explores differences in **training dynamics, representation structure, and model behavior**.

## Models

Two pretrained architectures are evaluated:

- **ResNet-18** – classic convolutional neural network
- **ViT-B/16** – Vision Transformer with 16×16 patches

Both models are adapted for new classification tasks by replacing the final classification layer.

## Datasets

Experiments are performed on multiple datasets:

- CIFAR-10
- CIFAR-100
- MNIST
- EMNIST

All images are resized to **224×224** and normalized using **ImageNet statistics** to match pretrained model requirements.

## Training Pipeline

A unified PyTorch pipeline is implemented to train and evaluate both architectures.

Key components include:

- dataset loading and preprocessing
- model initialization
- training loop
- evaluation metrics

The same pipeline allows consistent comparison between CNN and transformer models.

## Neural Tangent Kernel (NTK) Analysis

The project includes an empirical **Neural Tangent Kernel (NTK)** analysis to study training dynamics.

The NTK Gram matrix is computed by:

1. computing parameter gradients for model outputs
2. constructing the Jacobian matrix
3. computing pairwise inner products between gradients

This analysis provides insight into how model parameters influence learning behavior.

## Signal Propagation and Rank Analysis

To understand representation structure inside the networks, intermediate activations are extracted and analyzed.

The notebook computes:

- activation matrices
- singular value distributions
- **effective rank of feature representations**

This helps compare how CNNs and Vision Transformers compress or propagate information across layers.

## Experiments

Experiments are executed through a configurable experiment runner that trains models on different datasets.

Example experiment configuration:

```
("resnet18", "cifar10")
("vit_b16", "cifar10")
("resnet18", "cifar100")
("vit_b16", "cifar100")
```

## Notebook

Main experiment notebook:


CNNs_VITs.ipynb


## Environment

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
