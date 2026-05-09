# ADGNet: Attention Augmented Depthwise Separable GAN for Image Dehazing

## Overview

This repository contains the implementation and project materials for **ADGNet**, an efficiency-oriented Generative Adversarial Network designed for single image dehazing. The project focuses on restoring clear images from hazy or smoke-degraded inputs while keeping the model lightweight enough for practical real-time deployment.

Haze, fog, and smoke reduce image visibility by lowering contrast, fading colors, and hiding important scene details. This becomes especially important in surveillance, autonomous vehicles, robotics, smart monitoring systems, and emergency scenarios where poor visibility can affect both human observation and automated computer vision tasks.

ADGNet addresses this problem by combining **depthwise separable convolutions**, **squeeze-and-excitation attention blocks**, **residual learning**, and a **PatchGAN discriminator**. The model is trained and evaluated on the RESIDE OTS dataset and achieves competitive restoration quality with low computational complexity.

---

## Project Title

**ADGNet: Attention Augmented Depthwise Separable Generative Adversarial Network for Image Dehazing**

---

## Key Results

ADGNet was evaluated on the **RESIDE Outdoor Training Set (OTS)** benchmark.

| Metric | Value |
|---|---:|
| PSNR | 35.08 dB |
| SSIM | 0.9722 |
| Parameters | 1.60M |
| FLOPs | 15.74G |
| Inference Speed | 80.30 FPS on NVIDIA T4 |

These results show that ADGNet provides a strong balance between restoration quality and computational efficiency.

---

## Main Features

### Depthwise Separable Convolutions
ADGNet replaces standard convolutions in key parts of the network with depthwise separable convolutions. This reduces the number of parameters and computational cost while preserving useful feature extraction ability.

### Squeeze-and-Excitation Attention
SE blocks are used inside residual processing units to recalibrate feature channels. This helps the network emphasize informative image features and suppress less relevant responses.

### U-Net Inspired Generator
The generator follows an encoder–bottleneck–decoder structure with skip connections. This helps preserve spatial details while reconstructing clear images from hazy inputs.

### PatchGAN Discriminator
The discriminator evaluates local image patches instead of producing only a single global real/fake score. This encourages sharper textures, better local contrast, and improved edge restoration.

### Efficient Training Strategy
The training process uses stabilization techniques such as label smoothing, gradient clipping, asymmetric generator/discriminator updates, and separate learning rates for the generator and discriminator.

---

## Dataset

The model is trained and evaluated using the **Outdoor Training Set (OTS)** from the **RESIDE** dataset.

The dataset contains paired hazy and clear images, making it suitable for supervised image dehazing. During preprocessing, images are resized to `256 × 256` pixels and normalized to the range `[-1, 1]`.

---

## Model Architecture

ADGNet consists of two main components:

### Generator
- Initial `7 × 7` convolution layer
- Two depthwise separable downsampling blocks
- Eight residual SE blocks in the bottleneck
- Two transposed convolution upsampling layers
- Skip connections for spatial detail preservation
- Final `Tanh` activation for output scaling

### Discriminator
- PatchGAN-based architecture
- Initial convolution layer
- Five depthwise separable convolution blocks
- Instance normalization, LeakyReLU activation, and dropout
- Outputs a local patch-level realism map

---

## Training Configuration

| Setting | Value |
|---|---:|
| Framework | PyTorch |
| Training GPU | NVIDIA RTX 6000 Ada |
| Inference GPU | NVIDIA T4 |
| Epochs | 35 |
| Batch Size | 4 |
| Generator Learning Rate | 1e-4 |
| Discriminator Learning Rate | 4e-4 |
| Optimizer | Adam |
| Beta Values | β1 = 0.5, β2 = 0.999 |
| Scheduler | Cosine Annealing |

The total loss combines adversarial loss, L1 reconstruction loss, and perceptual consistency loss.

---

## Performance Across Haze Densities

| Haze Density | PSNR | SSIM |
|---|---:|---:|
| Light | 36.20 dB | 0.9835 |
| Medium | 35.08 dB | 0.9722 |
| Heavy | 33.72 dB | 0.9583 |

ADGNet remains stable across different haze levels, showing its usefulness for practical image restoration scenarios.

---

## Applications

ADGNet can be useful in:

- Surveillance and security systems
- Fire and smoke monitoring
- Autonomous vehicles
- Robotics vision systems
- Smart city monitoring
- Outdoor image restoration
- Preprocessing for object detection and tracking

---