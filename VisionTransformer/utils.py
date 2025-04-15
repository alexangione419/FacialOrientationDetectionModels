import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms


def visualize_patches(image: torch.Tensor, patch_size: int = 16) -> plt.Figure:
    """Visualize how an image is split into patches."""
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()

    # Create a grid to show patches
    h, w = image.shape[:2]
    patches = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    # Create a grid of patches
    n = int(np.sqrt(len(patches)))
    fig, axs = plt.subplots(n, n, figsize=(10, 10))

    for idx, patch in enumerate(patches):
        i, j = idx // n, idx % n
        axs[i, j].imshow(patch)
        axs[i, j].axis('off')

    plt.tight_layout()
    return fig


def visualize_attention(model: nn.Module, image: torch.Tensor, head_idx: int = 0, layer_idx: int = 0) -> plt.Figure:
    """Visualize attention weights for a specific head and layer."""
    model.eval()
    with torch.no_grad():
        # Get attention weights
        B = 1
        x = image.unsqueeze(0)

        # Forward pass through patch embedding
        x = model.patch_embed(x)

        # Add CLS token and position embeddings
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed

        # Get attention weights from specific layer and head
        for i, block in enumerate(model.blocks):
            if i == layer_idx:
                # Compute QKV
                qkv = block.attn.qkv(block.norm1(x))
                qkv = qkv.reshape(
                    B, -1, 3, block.attn.num_heads, block.attn.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Compute attention weights
                attn = (q @ k.transpose(-2, -1)) * \
                    (block.attn.head_dim ** -0.5)
                attn = attn.softmax(dim=-1)

                # Get weights for specified head
                attn_weights = attn[0, head_idx].cpu().numpy()
                break

    # Visualize attention weights
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn_weights, cmap='viridis')
    ax.set_title(f'Attention weights (Layer {layer_idx}, Head {head_idx})')
    plt.colorbar(im)

    return fig


def test_model(model: nn.Module, image_path: str, transform: transforms.Compose) -> tuple[int, float]:
    """Test the model on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        confidence = prob[0, pred].item()

    return pred, confidence


def inspect_positional_embeddings(model: nn.Module) -> plt.Figure:
    """Visualize the learned positional embeddings."""
    pos_embed = model.pos_embed.detach().cpu().numpy()[
        0, 1:]  # Exclude CLS token

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(pos_embed)

    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('Positional Embedding Correlations')
    plt.colorbar(im)

    return fig


def compute_attention_rollout(model: nn.Module, image: torch.Tensor) -> plt.Figure:
    """
    Compute and visualize attention rollout across all layers.
    This shows the cumulative effect of attention through the network.
    """
    model.eval()
    with torch.no_grad():
        B = 1
        x = image.unsqueeze(0)

        # Forward pass through patch embedding
        x = model.patch_embed(x)

        # Add CLS token and position embeddings
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed

        # Storage for attention weights from all layers
        attention_weights = []

        # Collect attention weights from all layers
        for block in model.blocks:
            # Get normalized input for attention
            norm_x = block.norm1(x)

            # Get QKV projections
            qkv = block.attn.qkv(norm_x)
            qkv = qkv.reshape(B, -1, 3, block.attn.num_heads,
                              block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Compute attention weights
            attn = (q @ k.transpose(-2, -1)) * (block.attn.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)

            # Average attention across heads
            attn_averaged = attn.mean(dim=1)
            attention_weights.append(attn_averaged)

            # Update x using the block's forward pass
            x = block(x)

        # Compute rollout through multiplication of attention weights
        rollout = torch.eye(
            attention_weights[0].shape[-1]).unsqueeze(0).to(image.device)
        for attn in attention_weights:
            rollout = torch.bmm(attn, rollout)

        # Get attention from CLS token to patches (first row)
        rollout = rollout[0, 0, 1:]  # Remove CLS token and get first row

        # Reshape to square for visualization
        size = int(np.sqrt(len(rollout)))
        attention_map = rollout.reshape(size, size).cpu().numpy()

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot original image
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        ax1.imshow(img_np)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Plot attention rollout
        im = ax2.imshow(attention_map, cmap='viridis')
        ax2.set_title('Attention Rollout')
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        return fig


def compute_average_attention_rollout(model: nn.Module, image_paths: list[str], transform: transforms.Compose, batch_size: int = 4) -> plt.Figure:
    """
    Compute and visualize average attention rollout across multiple images.
    This shows where the model generally focuses its attention across the dataset.

    Args:
        model: The Vision Transformer model
        image_paths: List of paths to images to analyze
        transform: Transformation pipeline to apply to images
        batch_size: Number of images to process at once
    """
    model.eval()
    accumulated_rollout = None
    num_images = len(image_paths)

    with torch.no_grad():
        # Process images in batches
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_size_actual = len(batch_paths)

            # Load and preprocess batch of images
            batch_images = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)

            # Stack images into a batch
            x = torch.stack(batch_images)

            # Forward pass through patch embedding
            x = model.patch_embed(x)

            # Add CLS token and position embeddings
            cls_tokens = model.cls_token.expand(batch_size_actual, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + model.pos_embed[:batch_size_actual]

            # Storage for attention weights from all layers
            attention_weights = []

            # Collect attention weights from all layers
            for block in model.blocks:
                # Get normalized input for attention
                norm_x = block.norm1(x)

                # Get QKV projections
                qkv = block.attn.qkv(norm_x)
                qkv = qkv.reshape(batch_size_actual, -1, 3,
                                  block.attn.num_heads, block.attn.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Compute attention weights
                attn = (q @ k.transpose(-2, -1)) * \
                    (block.attn.head_dim ** -0.5)
                attn = attn.softmax(dim=-1)

                # Average attention across heads
                attn_averaged = attn.mean(dim=1)
                attention_weights.append(attn_averaged)

                # Update x using the block's forward pass
                x = block(x)

            # Compute rollout for this batch
            batch_rollout = torch.eye(
                attention_weights[0].shape[-1]).unsqueeze(0).repeat(batch_size_actual, 1, 1).to(x.device)
            for attn in attention_weights:
                batch_rollout = torch.bmm(attn, batch_rollout)

            # Get attention from CLS token to patches (first row) for each image
            # Remove CLS token and get first row
            batch_rollout = batch_rollout[:, 0, 1:]

            # Accumulate rollout
            if accumulated_rollout is None:
                accumulated_rollout = batch_rollout.sum(dim=0)
            else:
                accumulated_rollout += batch_rollout.sum(dim=0)

    # Compute average
    average_rollout = accumulated_rollout / num_images

    # Reshape to square for visualization
    size = int(np.sqrt(len(average_rollout)))
    attention_map = average_rollout.reshape(size, size).cpu().numpy()

    # Create visualization with multiple subplots
    fig = plt.figure(figsize=(20, 10))

    # Plot attention heatmap
    ax1 = plt.subplot(121)
    im = ax1.imshow(attention_map, cmap='viridis')
    ax1.set_title('Average Attention Map')
    plt.colorbar(im, ax=ax1)

    # Plot attention overlay on sample image
    ax2 = plt.subplot(122)
    # Use the first image as reference for overlay
    sample_img = Image.open(image_paths[0]).convert('RGB')
    sample_img = transform(sample_img)
    img_np = sample_img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Create attention overlay
    attention_map_resized = torch.nn.functional.interpolate(
        torch.tensor(attention_map)[None, None],
        size=img_np.shape[:2],
        mode='bilinear',
        align_corners=False
    )[0, 0].numpy()

    # Normalize attention map
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (
        attention_map_resized.max() - attention_map_resized.min())

    # Create a red heatmap overlay
    heatmap = np.zeros((*img_np.shape[:2], 4))
    heatmap[..., 0] = 1  # Red channel
    heatmap[..., 3] = attention_map_resized  # Alpha channel

    ax2.imshow(img_np)
    ax2.imshow(heatmap)
    ax2.set_title('Attention Overlay on Sample Image')
    ax2.axis('off')

    plt.tight_layout()
    return fig
