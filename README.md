# major-project-2025
Real-Time Smoke Removal and Human Detection for Smart Surveillance in Fire Emergencies

DS-GAN: A Depthwise Separable GAN with Attention for Image Dehazing

Surveillance footage, especially during fire emergencies, is often compromised by haze and smoke, which reduce visibility and hinder both human and automated monitoring. This project introduces DS-GAN, a deep learning model that utilizes depthwise separable convolutions and squeeze-and-excitation (SE) attention blocks to efficiently remove haze from images. The architecture is designed to be lightweight and suitable for real-time applications, making it ideal for deployment in surveillance, autonomous vehicles, and robotics.

DS-GAN is trained and evaluated on the RESIDE OTS synthetic dataset and demonstrates strong performance, achieving a Peak Signal-to-Noise Ratio (PSNR) of 35.08 dB and a Structural Similarity Index (SSIM) of 0.9722. The model’s U-Net-inspired generator leverages modular residual blocks and SE channel attention, while the PatchGAN-based discriminator ensures the restoration of fine details and sharp edges.

Key features include:

Depthwise Separable Convolutions: Reduce computation and parameters while preserving high restoration quality.
Squeeze-and-Excitation Blocks: Adaptive channel-wise attention for enhanced feature representation.
PatchGAN Discriminator: Enforces local realism for sharper and more convincing outputs.
Efficient Training and Inference: Employs label smoothing, gradient clipping, and inference optimizations (FP16, TorchScript, CUDA).
DS-GAN achieves competitive dehazing results with fewer resources than many state-of-the-art methods, making it practical for real-world deployments. The model is robust across different haze densities and preserves color, detail, and contrast in dehazed images.

Future Work:
Ongoing research aims to further optimize DS-GAN for real-time video streams and evaluate its integration with object detection and tracking tasks in challenging environmental conditions.
