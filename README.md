# Denoising Diffusion Probabilistic Model (DDPM) on MNIST

This project is a PyTorch implementation of the research paper "Denoising Diffusion Probabilistic Models" by Ho et al., applied to the MNIST dataset of handwritten digits. The goal is to build and train a generative model from scratch that can synthesize high-quality images of digits from pure random noise.

### Key Hyperparameters

The training process was configured with the following parameters:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Timesteps (T)** | `1000` | Total number of diffusion steps. |
| **Image Size** | `32x32` | MNIST images were resized to fit the U-Net. |
| **Batch Size** | `128` | Number of images processed per training step. |
| **Epochs** | `100` | Total training passes over the dataset. |
| **Learning Rate** | `0.001` | Optimizer learning rate. |
| **Beta Schedule** | Linear | $\beta_t$ increases linearly from `0.0001` to `0.02`. |
| **Optimizer** | Adam | |
