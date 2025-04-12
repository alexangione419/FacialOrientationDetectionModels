import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample during training."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob

        # Create binary mask for dropping paths
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor

        return output


class PatchEmbedding(nn.Module):
    """Converts input images into patches and projects them into an embedding space."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection layer implemented ads a convolution
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width) -> (batch_size, n_patches, embed_dim)
        x = self.proj(x)
        x = self.flatten(x).transpose(1, 2)

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for processing patch embeddings."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Combined projection for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the multi-head attention mechanism."""
        batch_size, num_patches, embed_dim = x.shape

        # Split into heads and separate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3,
                                  self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Combine attention results and project
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron used in Transformer blocks."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with enhanced regularization."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer block."""
        # Apply attention block with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # Apply MLP block with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for face detection."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Learnable classification token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Create drop path schedule (increasing rate for deeper layers)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])

        # Final layer normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using a normal distribution."""
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m: nn.Module):
        """Initialize the weights of different layer types."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Vision Transformer."""
        # Create patch embeddings
        x = self.patch_embed(x)
        batch_size = x.shape[0]

        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract CLS token and classify
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x
