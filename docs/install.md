# Installation

## 1. Base Environment (PyTorch, nerfstudio and gsplat)

First, create and activate a new Conda environment:

```bash
# Create a conda environment
conda create --name trisafe -y python=3.9
conda activate trisafe

# tiny-cuda-nn requires > gcc 9.0
conda install -c conda-forge gxx=9.5.0

# If you do not have CUDA 11.8 locally, install it with the following command:
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install PyTorch and TorchVision
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install the base requirements
pip install -r requirements.txt
```

## 2. nuPlan Devkit

```bash
cd nuplan-devkit
pip install -r requirements_torch.txt
pip install -r requirements.txt
pip install -e .
cd ..
```

> **Note:** You also need to install COLMAP. Please get COLMAP from the [Official Installation Guide](https://colmap.github.io/install.html).

## 3. Data Processing Dependencies

Install dependencies for depth estimation, LiDAR point cloud processing, and image segmentation.

```bash
# Install requirements for nuplan data processing
pip install -r requirements_data.txt

# Install kiss-icp
# Note: This is a customized version with some changes, based on kiss-icp 1.0.0
# Tested on cmake 3.29.6 and gcc 9.5.0
cd thirdparty/kiss-icp
make editable
cd ../..

# Install UniDepth
# The requirements are aligned with MTGS
pip install -e thirdparty/UniDepth

# Download the pretrained models for UniDepth and Mask2Former
mkdir -p ckpts/huggingface

huggingface-cli download --local-dir ckpts/huggingface/facebook/mask2former-swin-large-cityscapes-semantic facebook/mask2former-swin-large-cityscapes-semantic

huggingface-cli download --local-dir ckpts/huggingface/lpiccinelli/unidepth-v2-vitl14 lpiccinelli/unidepth-v2-vitl14 --revision 1d0d3c52f60b5164629d279bb9a7546458e6dcc4
```
