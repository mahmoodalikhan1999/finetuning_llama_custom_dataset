# Fine-Tuning on a Personal Dataset 

This notebook demonstrates how to **fine-tune a pretrained deep learning model** on a **custom image dataset** using PyTorch. Ideal for tasks like image classification or domain adaptation when working with a small, personal dataset.

## 🗂️ Contents

- `Copy_of_finetuning_personal_dataset.ipynb`: Main notebook that loads a dataset, modifies a pretrained model (like ResNet or ViT), fine-tunes it, and evaluates results.
- Optional logic for data augmentation, transfer learning, and metrics.

## 🛠️ Features

- Uses PyTorch and torchvision
- Easily adaptable to different datasets (just update path and class labels)
- Supports GPU training
- Includes validation loop and performance visualization

## 🧰 Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- Jupyter

Install dependencies:

```bash
pip install -r requirements.txt

