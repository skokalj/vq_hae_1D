# Compressing Radio-Frequency Signals with Hierarchical Quantized Autoencoders

This repository is the code accompanying the paper [Deep-Learned Compression for Radio-Frequency Signal Classification](https://arxiv.org/abs/2403.03150). 

It is maintained by the Rowan University Artificial Intelligence Lab for Signals and Systems.

## Requirements:

1. [TorchDSP/torchsig](https://github.com/TorchDSP/torchsig)
2. torch, torchvision, matplotlib, torchsummary

## Installation

To get started with this repository, follow these instructions:

1. Clone the repository:
   ```bash
   git clone https://github.com/skokalj/vq_hae_1D.git
   cd vq_hae_1D
    ```
2. Install requirements
    ```bash
    pip install -r requirements.txt
    cd torchsig
    pip install .
    ```

## Citations

Code from the following repository was used in this research project:

1. **Hierarchical Quantized Autoencoders (HQA)**  
   Repository: [speechmatics/hqa](https://github.com/speechmatics/hqa)