# Pneumonia Detection from Chest X-Rays

A deep learning model built with **PyTorch** to classify pediatric chest X-ray images as **Normal** or **Pneumonia**.

## Overview
This project develops a convolutional neural network (CNN) for binary classification of chest X-ray images to detect pneumonia. It addresses class imbalance in medical imaging data using techniques like weighted loss and batch normalization.

- **Dataset**: ~8,000 pediatric chest X-ray images (from Kaggle's [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset)
- **Framework**: PyTorch
- **Model**: Custom CNN architecture (trained from scratch or fine-tuned)
- **Performance**: Test accuracy >90% (on hold-out set)

## Features
- Handles class imbalance with **Weighted Cross-Entropy Loss**
- Uses **Batch Normalization** for faster and stable training
- Data augmentation (transforms) for better generalization
- Simple inference script for single-image prediction
