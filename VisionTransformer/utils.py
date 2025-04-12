import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_patches(image, patch_size=16):
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


def visualize_attention(model, image, head_idx=0, layer_idx=0):
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


def test_model(model, image_path, transform):
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


def inspect_positional_embeddings(model):
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
