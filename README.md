# FacialOrientationDetectionModels


## Vision Transformer (ViT) Face Detector

A PyTorch implementation of a Vision Transformer (ViT) model for face detection in images. This implementation divides input images into patches, processes them through a transformer architecture, and performs binary classification to detect forward-facing faces.

### Features

- Vision Transformer architecture from scratch
- Patch-based image processing (16x16 patches)
- Multi-head self-attention mechanism
- Positional embeddings for spatial information
- Binary classification (face/no-face)
- Visualization tools for attention weights and patches
- Training with validation support

### Model Architecture

The Vision Transformer consists of:
1. Patch embedding layer
2. CLS token for classification
3. Learnable positional embeddings
4. Transformer encoder blocks with:
   - Multi-head self-attention
   - Layer normalization
   - MLP blocks
5. Classification head
ds
## Data Visualization

1. Open [web_interface/index.html](web_interface/index.html) in your browser.
2. Enjoy!
