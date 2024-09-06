# RotNet

**RotNet** is a self-supervised learning framework designed to leverage rotational pretext tasks to enhance feature extraction and improve digit classification on the MNIST dataset. By training a convolutional neural network (CNN) to predict image rotations, RotNet demonstrates how pretext tasks can be used effectively for transfer learning.

## Overview

RotNet follows these key steps:
1. **Pretext Task Training**: A CNN is trained to predict the rotation angle of images in the MNIST dataset, creating a self-supervised learning task.
2. **Fine-Tuning**: The model is adapted for digit classification by replacing the final layer and retraining it on the MNIST dataset.
3. **Evaluation**: The performance of the fine-tuned model is assessed on a test set to measure classification accuracy.

## Features

- **Self-Supervised Learning**: Utilizes a pretext task (image rotation prediction) to train the model.
- **Transfer Learning**: Fine-tunes the model on the MNIST digit classification task.
- **Progress Tracking**: Includes `tqdm` for monitoring training and evaluation progress.

## Project Structure

- **`src/dataloader.py`**: Contains data preparation and loading functions.
- **`src/model.py`**: Defines the convolutional neural network (CNN) architecture.
- **`src/train.py`**: Handles training, fine-tuning, and evaluation of the model.
- **`main.py`**: The entry point to run the full pipeline.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shouvik27/Advanced-Deep-Learning.git
   cd Advanced-Deep-Learning
