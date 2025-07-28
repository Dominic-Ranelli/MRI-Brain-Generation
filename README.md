# VAE-GAN with Attention for Medical Image Generation

A PyTorch implementation of a Variational Autoencoder with a Generative Adversarial Network (VAE-GAN) architecture, designed to generate and enhance high-quality medical images using attention mechanisms.

## Motivation 

This project combines the strengths of VAE and GANs to generate high-resolution, diverse, and realistic brain scan images. It's part of a research initiative aimed at addressing “limited access to diverse neuroimaging datasets”, especially for diseases like Alzheimer's.

# How it works:
1. Input and Encoder
— The model takes in real medical images (e.g. brain scans).
— Convolutional encoder compresses each image into a lower-dimensional latent vector z.
— Encoder outputs a mean (μ) and log-variance (log(σ²)).
— The model samples from latent space using the reparameterization trick, ensuring the latent distribution remains differentiable.

2. Decoder / Generator
— The sampled latent vector z is passed through a decoder, which attempts to reconstruct the original image.
— This decoder functions as the generator in GAN component, producing fake images to challenge the discriminator.
— The architecture includes channel and spatial attention mechanisms to help the model focus on important areas (e.g., tissue structures in medical images).

3. Discriminator
— Separate convolutional network (discriminator) is trained to tell apart:
	— Real medical images from the dataset
	— Reconstructed (fake) images from the decoder
— Feedback pushes the generator to produce sharper and more realistic images.

4. Loss Functions
— Reconstruction Loss: Measures pixel-level accuracy between original and reconstructed images
	— KL Divergence: Forces the latent space to follow a standard normal distribution.
	— Adversarial Loss: Rewards the generator when it fools the discriminator.
— Combined in a weighted total loss, allowing control over how much each part contributes.

## Features
— Variational Autoencoder (VAE) + GAN hybrid

— Attention modules (channel/spatial-wise)

— Custom logging to file and terminal

— Configurable training loop

— GPU support with PyTorch

— Compatible with Optuna for hyperparameter tuning

— Modular training and output saving

## Technologies Used
— Python 3

— PyTorch

— Torchvision

— PIL (Python Imagine Library)

— Matplotlib

— Optuna

— Datetime

— OS and Sys

— CUDA 

— Self Attention Mechanism

— Wasserstein GAN Loss
