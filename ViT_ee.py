import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from typing import Optional, Tuple
from CustomDataset import Data_prep_224_normal_N
from tqdm import tqdm
import torch.optim as optim
import torchvision.models as models
from collections import OrderedDict
import pytorch_lightning as pl
import time
import urllib.request
import zipfile
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights

class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings.

    For ViT-16, this splits a 224x224 image into 14x14 = 196 patches of size 16x16.
    Each patch is then linearly projected to the embedding dimension.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768):

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 196 patches for 224x224 image

        # Linear projection of flattened patches
        # Each patch is 16x16x3 = 768 values, projected to embed_dim
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Patch embeddings of shape (batch_size, n_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input size must be {self.img_size}x{self.img_size}"

        # Apply convolution to create patches and project them
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism as described in "Attention Is All You Need".

    This allows the model to attend to different parts of the sequence simultaneously
    from different representation subspaces.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            n_heads: int = 12,
            dropout: float = 0.1):

        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim) for scaled dot-product attention

        # Linear projections for queries, keys, and values
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape

        # Generate Q, K, V matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, n_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()  # GELU activation as used in the original paper
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MLP_EE(nn.Module):
    def __init__(self, input_dim: int = 768, mlp_ratio: float = 4.0,  output_dim: int = 10, dropout: float = 0.1):
        super().__init__()

        hidden_dim = int(input_dim * mlp_ratio)

        # Three dense (linear) layers
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, input_dim)
        self.dense3 = nn.Linear(input_dim, output_dim)

        self.gelu = nn.GELU()

        # Two dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        x = self.dense3(x)

        return x


class MLP_CNN_ignore(nn.Module):
    def __init__(self, input_dim: int = 784, mlp_ratio: float = 4.0,  output_dim: int = 10, dropout: float = 0.1):
        super().__init__()

        hidden_dim = int(input_dim * mlp_ratio)

        # Three dense (linear) layers
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, input_dim)
        self.dense3 = nn.Linear(input_dim, output_dim)

        self.gelu = nn.GELU()

        # Two dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        x = self.dense3(x)

        return x


class CNN_ignore(nn.Module):
    def __init__(self, hidden_dim, image_size, patch_size):
        super(CNN_ignore, self).__init__()

        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size

        # Calculate dimensions
        self.channels = hidden_dim
        self.width = self.height = image_size // patch_size

        # Define layers
        self.conv2d = nn.Conv2d(
            in_channels=self.channels,
            out_channels=16,
            kernel_size=(3, 3),
            padding='same'
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        # Calculate output size after conv and maxpool for flattening
        self.pooled_width = self.width // 2
        self.pooled_height = self.height // 2
        self.flattened_size = 16 * self.pooled_width * self.pooled_height

    def forward(self, y):

        # Reshape from (batch_size, seq_len, channels) to (batch_size, height, width, channels)
        batch_size = y.size(0)
        y = y.view(batch_size, self.height, self.width, self.channels)

        # Convert from NHWC to NCHW format (PyTorch convention)
        y = y.permute(0, 3, 1, 2)  # Shape: (batch_size, channels, height, width)

        # Apply Conv2D with ELU activation
        y = F.elu(self.conv2d(y))

        # Apply MaxPooling
        y = self.maxpool(y)

        # Flatten
        y = torch.flatten(y, start_dim=1)

        return y


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
    1. Layer normalization
    2. Multi-head self-attention
    3. Residual connection
    4. Layer normalization
    5. MLP (feed-forward network)
    6. Residual connection

    Note: ViT uses pre-normalization (LayerNorm before attention/MLP)
    rather than post-normalization.
    """

    def __init__(self, embed_dim: int = 768, n_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization with residual connections
        x = x + self.attn(self.norm1(x))  # Attention block
        x = x + self.mlp(self.norm2(x))  # MLP block
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation based on:
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

    This implementation specifically follows the ViT-Base/16 configuration:
    - Patch size: 16x16
    - Embedding dimension: 768
    - Number of attention heads: 12
    - Number of transformer layers: 12
    - MLP hidden dimension: 3072 (4 * 768)
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            n_classes: int = 10,
            embed_dim: int = 768,
            depth: int = 12,
            n_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0, # was 0.1
            attention_dropout: float = 0.0 # was 0.1
    ):
        super().__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Learnable [CLS] token - used for classification
        # This token is prepended to the sequence and its final representation
        # is used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings - learnable position encodings
        # We need n_patches + 1 positions (for the CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, attention_dropout)
            for _ in range(depth)
        ])

        # layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head, use MLP as stated in the papers
        self.MLP_EE = MLP_EE(embed_dim, mlp_ratio, output_dim=n_classes, dropout=dropout)
        self.CNN_ignore = CNN_ignore(embed_dim, img_size, patch_size)
        self.MLP_CNN = MLP_CNN_ignore(784, mlp_ratio, output_dim=n_classes, dropout=dropout)

        # Final classification head - a single linear layer
        self.final_head = nn.Linear(embed_dim, n_classes)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights following the original paper."""
        # Initialize positional embeddings with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def extract_features(self, x):
        B = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Pass through transformer blocks
        # for block in self.blocks:
        #     x = block(x)
        x1 = self.blocks[0](x)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x2)
        x4 = self.blocks[3](x3)
        x5 = self.blocks[4](x4)
        x6 = self.blocks[5](x5)
        x7 = self.blocks[6](x6)
        x8 = self.blocks[7](x7)
        x9 = self.blocks[8](x8)
        x10 = self.blocks[9](x9)
        x11 = self.blocks[10](x10)
        x12 = self.blocks[11](x11)

        # Apply normalization
        x12 = self.norm(x12)

        return x12[:, 0]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extraction part of the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Feature tensor of shape (batch_size, embed_dim)
        """
        B = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Pass through transformer blocks
        # for block in self.blocks:
        #     x = block(x)
        x1 = self.blocks[0](x)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x2)
        x4 = self.blocks[3](x3)
        x5 = self.blocks[4](x4)
        x6 = self.blocks[5](x5)
        x7 = self.blocks[6](x6)
        x8 = self.blocks[7](x7)
        x9 = self.blocks[8](x8)
        x10 = self.blocks[9](x9)
        x11 = self.blocks[10](x10)
        x12 = self.blocks[11](x11)

        # Apply normalization
        x12 = self.norm(x12)
        x11 = self.norm(x11)
        x10 = self.norm(x10)
        x9 = self.norm(x9)
        x8 = self.norm(x8)
        x7 = self.norm(x7)
        x6 = self.norm(x6)
        x5 = self.norm(x5)
        x4 = self.norm(x4)
        x3 = self.norm(x3)
        x2 = self.norm(x2)
        x1 = self.norm(x1)
        # Return CLS token representation (first token)

        if head_type == "MLP_EE":
            return x12[:, 0], x11[:, 0], x10[:, 0], x9[:, 0], x8[:, 0], x7[:, 0], x6[:, 0]

        elif head_type == "CNN_ignore":
            return x12[:, 0], x9[:, 1:], x6[:, 1:], x3[:, 1:]
            # only returning the CLS token for x12 (which goes to final_head) and the patch tokens for the other layers (which go to CNN_ignore and MLP_CNN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if head_type == "MLP_EE":
            x12, x11, x10, x9, x8, x7, x6  = self.forward_features(x)
            x12 = self.final_head(x12)
            x11 = self.MLP_EE(x11)
            x10 = self.MLP_EE(x10)
            x9 = self.MLP_EE(x9)
            x8 = self.MLP_EE(x8)
            x7 = self.MLP_EE(x7)
            x6 = self.MLP_EE(x6)

            return x12, x11, x10, x9, x8, x7, x6

        elif head_type == "CNN_ignore":
            x12, x9, x6, x3 = self.forward_features(x)
            x12 = self.final_head(x12)
            x9 = self.MLP_CNN(self.CNN_ignore(x9))
            x6 = self.MLP_CNN(self.CNN_ignore(x6))
            x3 = self.MLP_CNN(self.CNN_ignore(x3))

            return x12, x9, x6, x3


def create_vit_base_16_ee(n_classes, pretrained: bool = False) -> VisionTransformer:

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=n_classes,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1
    )

    if pretrained:
        # In practice, you would load pretrained weights here
        print("Pretrained weights loading not implemented in this example")

    return model


def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    running_loss = 0.0
    total = 0

    # Initialize tracking variables based on head type
    if head_type == "MLP_EE":
        correct = {12: 0, 11: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0}
    elif head_type == "CNN_ignore":
        correct = {12: 0, 9: 0, 6: 0, 3: 0}

    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass - outputs depend on head_type
        if head_type == "MLP_EE":
            outputs = model(inputs)  # Returns 7 outputs
            output12, output11, output10, output9, output8, output7, output6 = outputs

            # Calculate loss for each output
            loss_12 = criterion(output12, targets)
            loss_11 = criterion(output11, targets)
            loss_10 = criterion(output10, targets)
            loss_9 = criterion(output9, targets)
            loss_8 = criterion(output8, targets)
            loss_7 = criterion(output7, targets)
            loss_6 = criterion(output6, targets)

            # Weighted loss combination
            loss = 2 * loss_12 + loss_11 + loss_10 + loss_9 + loss_8 + loss_7 + loss_6

        elif head_type == "CNN_ignore":
            outputs = model(inputs)  # Returns 4 outputs
            output12, output9, output6, output3 = outputs

            # Calculate loss for each output
            loss_12 = criterion(output12, targets)
            loss_9 = criterion(output9, targets)
            loss_6 = criterion(output6, targets)
            loss_3 = criterion(output3, targets)

            # Weighted loss combination
            loss = 2 * loss_12 + loss_9 + loss_6 + loss_3

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item()
        total += targets.size(0)

        # Calculate accuracies based on head type
        if head_type == "MLP_EE":
            _, predicted_12 = output12.max(1)
            _, predicted_11 = output11.max(1)
            _, predicted_10 = output10.max(1)
            _, predicted_9 = output9.max(1)
            _, predicted_8 = output8.max(1)
            _, predicted_7 = output7.max(1)
            _, predicted_6 = output6.max(1)

            correct[12] += predicted_12.eq(targets).sum().item()
            correct[11] += predicted_11.eq(targets).sum().item()
            correct[10] += predicted_10.eq(targets).sum().item()
            correct[9] += predicted_9.eq(targets).sum().item()
            correct[8] += predicted_8.eq(targets).sum().item()
            correct[7] += predicted_7.eq(targets).sum().item()
            correct[6] += predicted_6.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc_6': 100. * correct[6] / total,
                'acc_9': 100. * correct[9] / total,
                'acc_12': 100. * correct[12] / total
            })

        elif head_type == "CNN_ignore":
            _, predicted_12 = output12.max(1)
            _, predicted_9 = output9.max(1)
            _, predicted_6 = output6.max(1)
            _, predicted_3 = output3.max(1)

            correct[12] += predicted_12.eq(targets).sum().item()
            correct[9] += predicted_9.eq(targets).sum().item()
            correct[6] += predicted_6.eq(targets).sum().item()
            correct[3] += predicted_3.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc_3': 100. * correct[3] / total,
                'acc_6': 100. * correct[6] / total,
                'acc_12': 100. * correct[12] / total
            })

    if scheduler:
        scheduler.step()

    # Return results based on head type
    if head_type == "MLP_EE":
        return (running_loss / len(dataloader),
                100. * correct[6] / total,
                100. * correct[7] / total,
                100. * correct[8] / total,
                100. * correct[9] / total,
                100. * correct[10] / total,
                100. * correct[11] / total,
                100. * correct[12] / total)
    elif head_type == "CNN_ignore":
        return (running_loss / len(dataloader),
                100. * correct[3] / total,
                100. * correct[6] / total,
                100. * correct[9] / total,
                100. * correct[12] / total)


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    total = 0

    # Initialize tracking variables based on head type
    if head_type == "MLP_EE":
        correct = {12: 0, 11: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0}
    elif head_type == "CNN_ignore":
        correct = {12: 0, 9: 0, 6: 0, 3: 0}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass based on head type
            if head_type == "MLP_EE":
                outputs = model(inputs)
                output12, output11, output10, output9, output8, output7, output6 = outputs

                # Calculate loss for each output
                loss_12 = criterion(output12, targets)
                loss_11 = criterion(output11, targets)
                loss_10 = criterion(output10, targets)
                loss_9 = criterion(output9, targets)
                loss_8 = criterion(output8, targets)
                loss_7 = criterion(output7, targets)
                loss_6 = criterion(output6, targets)

                # Weighted loss combination
                loss = 2 * loss_12 + loss_11 + loss_10 + loss_9 + loss_8 + loss_7 + loss_6

            elif head_type == "CNN_ignore":
                outputs = model(inputs)
                output12, output9, output6, output3 = outputs

                # Calculate loss for each output
                loss_12 = criterion(output12, targets)
                loss_9 = criterion(output9, targets)
                loss_6 = criterion(output6, targets)
                loss_3 = criterion(output3, targets)

                # Weighted loss combination
                loss = 2 * loss_12 + loss_9 + loss_6 + loss_3

            # Update statistics
            running_loss += loss.item()
            total += targets.size(0)

            # Calculate accuracies based on head type
            if head_type == "MLP_EE":
                _, predicted_12 = output12.max(1)
                _, predicted_11 = output11.max(1)
                _, predicted_10 = output10.max(1)
                _, predicted_9 = output9.max(1)
                _, predicted_8 = output8.max(1)
                _, predicted_7 = output7.max(1)
                _, predicted_6 = output6.max(1)

                correct[12] += predicted_12.eq(targets).sum().item()
                correct[11] += predicted_11.eq(targets).sum().item()
                correct[10] += predicted_10.eq(targets).sum().item()
                correct[9] += predicted_9.eq(targets).sum().item()
                correct[8] += predicted_8.eq(targets).sum().item()
                correct[7] += predicted_7.eq(targets).sum().item()
                correct[6] += predicted_6.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc_6': 100. * correct[6] / total,
                    'acc_9': 100. * correct[9] / total,
                    'acc_12': 100. * correct[12] / total
                })

            elif head_type == "CNN_ignore":
                _, predicted_12 = output12.max(1)
                _, predicted_9 = output9.max(1)
                _, predicted_6 = output6.max(1)
                _, predicted_3 = output3.max(1)

                correct[12] += predicted_12.eq(targets).sum().item()
                correct[9] += predicted_9.eq(targets).sum().item()
                correct[6] += predicted_6.eq(targets).sum().item()
                correct[3] += predicted_3.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc_3': 100. * correct[3] / total,
                    'acc_6': 100. * correct[6] / total,
                    'acc_12': 100. * correct[12] / total
                })

    # Return results based on head type
    if head_type == "MLP_EE":
        return (running_loss / len(dataloader),
                100. * correct[6] / total,
                100. * correct[7] / total,
                100. * correct[8] / total,
                100. * correct[9] / total,
                100. * correct[10] / total,
                100. * correct[11] / total,
                100. * correct[12] / total)
    elif head_type == "CNN_ignore":
        return (running_loss / len(dataloader),
                100. * correct[3] / total,
                100. * correct[6] / total,
                100. * correct[9] / total,
                100. * correct[12] / total)


# Logging utility
def log_message(message):
    log_dir = "running logs"
    log_file = os.path.join(log_dir, "VIT_EE running log")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(message + "\n")


def train(updated_ee_vit):
    # Training loop
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        log_message(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        if head_type == "MLP_EE":
            train_loss, train_acc_6, train_acc_7, train_acc_8, train_acc_9, train_acc_10, train_acc_11, train_acc_12 = train_epoch(
                updated_ee_vit, train_loader, criterion, optimizer, scheduler)
            train_log = (f"Train Loss: {train_loss:.4f} | Train Acc: "
                         f"6: {train_acc_6:.2f}%, 9: {train_acc_9:.2f}%, 12: {train_acc_12:.2f}%")
            print(train_log)
            log_message(train_log)
        elif head_type == "CNN_ignore":
            train_loss, train_acc_3, train_acc_6, train_acc_9, train_acc_12 = train_epoch(
                updated_ee_vit, train_loader, criterion, optimizer, scheduler)
            train_log = (f"Train Loss: {train_loss:.4f} | Train Acc: "
                         f"3: {train_acc_3:.2f}%, 6: {train_acc_6:.2f}%, 12: {train_acc_12:.2f}%")
            print(train_log)
            log_message(train_log)

        # Validate
        if head_type == "MLP_EE":
            val_loss, val_acc_6, val_acc_7, val_acc_8, val_acc_9, val_acc_10, val_acc_11, val_acc_12 = validate(
                updated_ee_vit, val_loader, criterion)
            val_log = (f"Val Loss: {val_loss:.4f} | Val Acc: "
                       f"6: {val_acc_6:.2f}%, 9: {val_acc_9:.2f}%, 12: {val_acc_12:.2f}%")
            print(val_log)
            log_message(val_log)
        elif head_type == "CNN_ignore":
            val_loss, val_acc_3, val_acc_6, val_acc_9, val_acc_12 = validate(
                updated_ee_vit, val_loader, criterion)
            val_log = (f"Val Loss: {val_loss:.4f} | Val Acc: "
                       f"3: {val_acc_3:.2f}%, 6: {val_acc_6:.2f}%, 12: {val_acc_12:.2f}%")
            print(val_log)
            log_message(val_log)

        # Save best model
        if val_acc_12 > best_val_acc:
            best_val_acc = val_acc_12
            torch.save(updated_ee_vit.state_dict(), f'weights/vit_ee_{dataset_type}_{head_type}_best.pth')
            save_log = f"Saved best model with acc: {best_val_acc:.2f}%"
            print(save_log)
            log_message(save_log)

    # Load best model for testing
    updated_ee_vit.load_state_dict(torch.load(f'weights/vit_ee_{dataset_type}_{head_type}_best.pth'))

    # Test
    if head_type == "MLP_EE":
        test_loss, test_acc_6, test_acc_7, test_acc_8, test_acc_9, test_acc_10, test_acc_11, test_acc_12 = validate(
            updated_ee_vit, test_loader, criterion)
        test_log = (f"\nTest Loss: {test_loss:.4f} | Test Acc: "
                    f"6: {test_acc_6:.2f}%, 9: {test_acc_9:.2f}%, 12: {test_acc_12:.2f}%")
        print(test_log)
        log_message(test_log)
    elif head_type == "CNN_ignore":
        test_loss, test_acc_3, test_acc_6, test_acc_9, test_acc_12 = validate(
            updated_ee_vit, test_loader, criterion)
        test_log = (f"\nTest Loss: {test_loss:.4f} | Test Acc: "
                    f"3: {test_acc_3:.2f}%, 6: {test_acc_6:.2f}%, 12: {test_acc_12:.2f}%")
        print(test_log)
        log_message(test_log)


def transfer_weights_from_vit_to_ee_vit(vit_model, ee_vit_model):
    # Dictionary for mapping keys between models
    weight_mapping = {
        # Patch embedding and tokens
        'conv_proj.weight': 'patch_embed.projection.weight',
        'conv_proj.bias': 'patch_embed.projection.bias',
        'class_token': 'cls_token',
        'encoder.pos_embedding': 'pos_embed',

        # Final head mapping for the linear layer
        'heads.weight': 'final_head.weight',
        'heads.bias': 'final_head.bias',

        # Transformer blocks mapping (for each layer)
        # Original ViT has 'encoder.layers.encoder_layer_X...'
        # EE ViT has 'blocks.X...'
    }

    # Add mappings for each transformer block (0-11)
    for i in range(12):
        # Layer norm 1
        weight_mapping[f'encoder.layers.encoder_layer_{i}.ln_1.weight'] = f'blocks.{i}.norm1.weight'
        weight_mapping[f'encoder.layers.encoder_layer_{i}.ln_1.bias'] = f'blocks.{i}.norm1.bias'

        # Self attention
        # Original ViT uses in_proj_weight/bias, EE ViT uses qkv.weight
        weight_mapping[
            f'encoder.layers.encoder_layer_{i}.self_attention.in_proj_weight'] = f'blocks.{i}.attn.qkv.weight'
        # Skip in_proj_bias as the ee_vit doesn't have qkv bias

        # Projection
        weight_mapping[
            f'encoder.layers.encoder_layer_{i}.self_attention.out_proj.weight'] = f'blocks.{i}.attn.proj.weight'
        weight_mapping[f'encoder.layers.encoder_layer_{i}.self_attention.out_proj.bias'] = f'blocks.{i}.attn.proj.bias'

        # Layer norm 2
        weight_mapping[f'encoder.layers.encoder_layer_{i}.ln_2.weight'] = f'blocks.{i}.norm2.weight'
        weight_mapping[f'encoder.layers.encoder_layer_{i}.ln_2.bias'] = f'blocks.{i}.norm2.bias'

        # MLP
        weight_mapping[f'encoder.layers.encoder_layer_{i}.mlp.0.weight'] = f'blocks.{i}.mlp.fc1.weight'
        weight_mapping[f'encoder.layers.encoder_layer_{i}.mlp.0.bias'] = f'blocks.{i}.mlp.fc1.bias'
        weight_mapping[f'encoder.layers.encoder_layer_{i}.mlp.3.weight'] = f'blocks.{i}.mlp.fc2.weight'
        weight_mapping[f'encoder.layers.encoder_layer_{i}.mlp.3.bias'] = f'blocks.{i}.mlp.fc2.bias'

    # Final norm layer
    weight_mapping['encoder.ln.weight'] = 'norm.weight'
    weight_mapping['encoder.ln.bias'] = 'norm.bias'

    # Transfer weights for the mapped keys
    state_dict_vit = vit_model.state_dict()
    state_dict_ee_vit = ee_vit_model.state_dict()

    transferred_params = 0
    skipped_params = []

    for vit_key, ee_vit_key in weight_mapping.items():
        if vit_key in state_dict_vit and ee_vit_key in state_dict_ee_vit:
            if state_dict_vit[vit_key].shape == state_dict_ee_vit[ee_vit_key].shape:
                state_dict_ee_vit[ee_vit_key].copy_(state_dict_vit[vit_key])
                transferred_params += 1
            else:
                print(
                    f"Shape mismatch: {vit_key} {state_dict_vit[vit_key].shape} vs {ee_vit_key} {state_dict_ee_vit[ee_vit_key].shape}")
                skipped_params.append((vit_key, ee_vit_key))
        else:
            if vit_key not in state_dict_vit:
                print(f"Key not found in source model: {vit_key}")
            if ee_vit_key not in state_dict_ee_vit:
                print(f"Key not found in target model: {ee_vit_key}")
            skipped_params.append((vit_key, ee_vit_key))

    # Special handling for self-attention weights (converting in_proj to qkv format)
    for i in range(12):
        vit_key = f'encoder.layers.encoder_layer_{i}.self_attention.in_proj_weight'
        vit_bias_key = f'encoder.layers.encoder_layer_{i}.self_attention.in_proj_bias'
        ee_vit_key = f'blocks.{i}.attn.qkv.weight'

        if vit_key in state_dict_vit and ee_vit_key in state_dict_ee_vit:
            # The in_proj_weight in ViT combines Q, K, V matrices
            # We need to adapt this for the ee_vit format
            in_proj_weight = state_dict_vit[vit_key]
            if in_proj_weight.shape[0] == 3 * state_dict_ee_vit[ee_vit_key].shape[0]:
                state_dict_ee_vit[ee_vit_key].copy_(in_proj_weight)
                transferred_params += 1
            else:
                print(
                    f"Shape mismatch for attention: {vit_key} {in_proj_weight.shape} vs {ee_vit_key} {state_dict_ee_vit[ee_vit_key].shape}")
                skipped_params.append((vit_key, ee_vit_key))

    # Update the ee_vit model with transferred weights
    ee_vit_model.load_state_dict(state_dict_ee_vit)

    print(f"Successfully transferred {transferred_params} parameters")
    print(f"Skipped {len(skipped_params)} parameter mappings")

    return ee_vit_model, skipped_params


def transfer_weights(source_model, target_model, custom_mappings=None):
    """
    Enhanced weight transfer function with custom parameter mappings
    """
    # Your existing transfer code
    transferred = 0
    skipped = 0

    # Add custom mappings for the attention blocks
    if custom_mappings is None:
        custom_mappings = {}
    for i in range(12):
        custom_mappings[
            f"encoder.layers.encoder_layer_{i}.self_attention.in_proj_weight"] = f"blocks.{i}.attn.qkv.weight"

    target_state_dict = target_model.state_dict()
    for name, param in source_model.state_dict().items():
        # Check custom mappings first
        if name in custom_mappings:
            target_name = custom_mappings[name]
            if target_name in target_state_dict and param.shape == target_state_dict[target_name].shape:
                target_state_dict[target_name].copy_(param)
                transferred += 1
            else:
                skipped += 1
        # Then try direct mapping
        elif name in target_state_dict and param.shape == target_state_dict[name].shape:
            target_state_dict[name].copy_(param)
            transferred += 1
        else:
            skipped += 1

    print(f"Successfully transferred {transferred} parameters")
    print(f"Skipped {skipped} parameter mappings")

    return target_model


def white_board_test(model, image_path, model_name):
    # Load and preprocess a single image for testing
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Define the same transforms used for the test dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()  # Keep a copy for display
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Optimal thresholds for the ViT_ee
    optimal_thresholds = thresholds = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    # Set model to evaluation mode
    model.eval()
    classified_label, exit_point = threshold_inference_single_image(model, image_tensor, optimal_thresholds)

    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title(f'{model_name} EE pattern on ee_ViT \n'
              f'label: {classified_label}, Exit at : {exit_point}')
    plt.show()


def VGG16_white_board_test():

    # VGG16 EE pattern
    white_board_test(updated_ee_vit, "Results/white_board/vgg16/3_cat/white_board_20250405_020418.png", "VGG16")
    white_board_test(updated_ee_vit, "Results/white_board/vgg16/5_dog/white_board_20250404_222037.png", "VGG16")
    white_board_test(updated_ee_vit, "Results/white_board/vgg16/6_frog/white_board_20250406_233829.png", "VGG16")
    white_board_test(updated_ee_vit, "Results/white_board/vgg16/9_truck/white_board_20250405_193221.png", "VGG16")


def Resnet50_white_board_test():

    # Resnet50 EE pattern
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/0_airplane/white_board_20250403_171932.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/1_automobile/500/white_board_20250403_192223.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/2_bird/white_board_20250403_200709.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/3_cat/white_board_20250429_160014.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/4_deer/white_board_20250404_001914.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/5_dog/white_board_20250404_011411.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/6_frog/white_board_20250404_120430.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/7_horse/white_board_20250404_131530.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/8_ship/white_board_20250404_153240.png", "Res50")
    white_board_test(updated_ee_vit, "Results/white_board/resnet50/9_truck/white_board_20250404_170636.png", "Res50")


def threshold_inference_single_image(model, image_tensor, thresholds):
    model.eval()
    with torch.no_grad():
        if head_type == "MLP_EE":
            output12, output11, output10, output9, output8, output7, output6 = model(image_tensor)

            # Apply softmax to get probabilities
            probs6 = torch.nn.functional.softmax(output6, dim=1)
            probs7 = torch.nn.functional.softmax(output7, dim=1)
            probs8 = torch.nn.functional.softmax(output8, dim=1)
            probs9 = torch.nn.functional.softmax(output9, dim=1)
            probs10 = torch.nn.functional.softmax(output10, dim=1)
            probs11 = torch.nn.functional.softmax(output11, dim=1)
            probs12 = torch.nn.functional.softmax(output12, dim=1)

            # Get max probability for each exit
            max_prob6, pred6 = torch.max(probs6, dim=1)
            max_prob7, pred7 = torch.max(probs7, dim=1)
            max_prob8, pred8 = torch.max(probs8, dim=1)
            max_prob9, pred9 = torch.max(probs9, dim=1)
            max_prob10, pred10 = torch.max(probs10, dim=1)
            max_prob11, pred11 = torch.max(probs11, dim=1)
            max_prob12, pred12 = torch.max(probs12, dim=1)

            # Early exit logic based on confidence
            if max_prob6 >= thresholds[0]:
                return pred6.item(), 6
            elif max_prob7 >= thresholds[1]:
                return pred7.item(), 7
            elif max_prob8 >= thresholds[2]:
                return pred8.item(), 8
            elif max_prob9 >= thresholds[3]:
                return pred9.item(), 9
            elif max_prob10 >= thresholds[4]:
                return pred10.item(), 10
            elif max_prob11 >= thresholds[5]:
                return pred11.item(), 11
            else:
                return pred12.item(), 12

        elif head_type == "CNN_ignore":
            output12, output9, output6, output3 = model(image_tensor)

            # Apply softmax to get probabilities
            probs3 = torch.nn.functional.softmax(output3, dim=1)
            probs6 = torch.nn.functional.softmax(output6, dim=1)
            probs9 = torch.nn.functional.softmax(output9, dim=1)
            probs12 = torch.nn.functional.softmax(output12, dim=1)

            # Get max probability for each exit
            max_prob3, pred3 = torch.max(probs3, dim=1)
            max_prob6, pred6 = torch.max(probs6, dim=1)
            max_prob9, pred9 = torch.max(probs9, dim=1)
            max_prob12, pred12 = torch.max(probs12, dim=1)

            print(max_prob3)
            # Early exit logic based on confidence
            if max_prob3 >= thresholds[0]:
                return pred3.item(), 3
            elif max_prob6 >= thresholds[1]:
                return pred6.item(), 6
            elif max_prob9 >= thresholds[2]:
                return pred9.item(), 9
            else:
                return pred12.item(), 12


def confidence(output, confidence_type):
    if confidence_type == "entropy":
        probs = F.softmax(output, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        pred = probs.argmax(dim=1).item()
        return entropy, pred

    elif confidence_type == "max_prob":
        probs = F.softmax(output, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        return max_prob, pred


def threshold_inference(model, test_loader, thresholds):
    model.eval()

    total = 0
    all_preds = []
    all_labels = []

    if head_type == "MLP_EE":
        correct = {12: 0, 11: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0}
        exit_counts = {12: 0, 11: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0}
    elif head_type == "CNN_ignore":
        correct = {12: 0, 9: 0, 6: 0, 3: 0}
        exit_counts = {12: 0, 9: 0, 6: 0, 3: 0}

    inference_times = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            total += len(labels)
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            batch_preds = []
            batch_exits = []

            for i in range(len(images)):
                img = images[i].unsqueeze(0)

                # Process based on head type
                if head_type == "MLP_EE":
                    output12, output11, output10, output9, output8, output7, output6 = model(img)

                    # Process exits in order (6, 7, 8, 9, 10, 11, 12)
                    exit_point = 12  # Default exit point (last exit)
                    pred = None

                    # Check exit6 output
                    if output6 is not None:
                        probs6 = F.softmax(output6, dim=1)
                        entropy6 = -torch.sum(probs6 * torch.log(probs6 + 1e-10), dim=1)
                        if entropy6.item() < thresholds[0]:  # Exit at 6
                            exit_point = 6
                            pred = probs6.argmax(dim=1).item()

                    # Check exit7 output if didn't exit at 6
                    if exit_point == 12 and output7 is not None:
                        probs7 = F.softmax(output7, dim=1)
                        entropy7 = -torch.sum(probs7 * torch.log(probs7 + 1e-10), dim=1)
                        if entropy7.item() < thresholds[0]:  # Use same threshold or adjust as needed
                            exit_point = 7
                            pred = probs7.argmax(dim=1).item()

                    # Check exit8 output if didn't exit earlier
                    if exit_point == 12 and output8 is not None:
                        probs8 = F.softmax(output8, dim=1)
                        entropy8 = -torch.sum(probs8 * torch.log(probs8 + 1e-10), dim=1)
                        if entropy8.item() < thresholds[0]:  # Use same threshold or adjust as needed
                            exit_point = 8
                            pred = probs8.argmax(dim=1).item()

                    # Check exit9 output if didn't exit earlier
                    if exit_point == 12 and output9 is not None:
                        probs9 = F.softmax(output9, dim=1)
                        entropy9 = -torch.sum(probs9 * torch.log(probs9 + 1e-10), dim=1)
                        if entropy9.item() < thresholds[1]:  # Exit at 9
                            exit_point = 9
                            pred = probs9.argmax(dim=1).item()

                    # Check exit10 output if didn't exit earlier
                    if exit_point == 12 and output10 is not None:
                        probs10 = F.softmax(output10, dim=1)
                        entropy10 = -torch.sum(probs10 * torch.log(probs10 + 1e-10), dim=1)
                        if entropy10.item() < thresholds[1]:  # Use same threshold as exit9
                            exit_point = 10
                            pred = probs10.argmax(dim=1).item()

                    # Check exit11 output if didn't exit earlier
                    if exit_point == 12 and output11 is not None:
                        probs11 = F.softmax(output11, dim=1)
                        entropy11 = -torch.sum(probs11 * torch.log(probs11 + 1e-10), dim=1)
                        if entropy11.item() < thresholds[2]:  # Use threshold for later exits
                            exit_point = 11
                            pred = probs11.argmax(dim=1).item()

                    # If no early exit, use the final exit (12)
                    if exit_point == 12:
                        probs12 = F.softmax(output12, dim=1)
                        pred = probs12.argmax(dim=1).item()

                elif head_type == "CNN_ignore":
                    output12, output9, output6, output3 = model(img)

                    # Process exits in order (3, 6, 9, 12)
                    exit_point = 12  # Default exit point (last exit)
                    pred = None

                    # Check exit3 output
                    confidence3, pred_label = confidence(output3, "max_prob")
                    if confidence3.item() > thresholds[0]:  # Exit at 3
                        exit_point = 3
                        pred = pred_label

                    # Check exit6 output if didn't exit earlier
                    if exit_point == 12:
                        confidence6, pred_label = confidence(output6, "max_prob")
                        if confidence6.item() > thresholds[1]:  # Exit at 6
                            exit_point = 6
                            pred = pred_label

                    # Check exit9 output if didn't exit earlier
                    if exit_point == 12:
                        confidence9, pred_label = confidence(output9, "max_prob")
                        if confidence9.item() > thresholds[2]:  # Exit at 9
                            exit_point = 9
                            pred = pred_label

                    # If no early exit, use the final exit (12)
                    if exit_point == 12:
                        probs12 = F.softmax(output12, dim=1)
                        pred = probs12.argmax(dim=1).item()

                batch_preds.append(pred)
                batch_exits.append(exit_point)
                exit_counts[exit_point] += 1

                # Check if prediction is correct
                if pred == labels[i].item():
                    correct[exit_point] += 1

            end_time = time.time()
            inference_times.append(end_time - start_time)

            all_preds.extend(batch_preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    overall_correct = sum(correct.values())
    overall_accuracy = 100.0 * overall_correct / total

    # Individual exit accuracies
    exit_accuracy = {}
    if head_type == "MLP_EE":
        exit_points = [6, 7, 8, 9, 10, 11, 12]
    elif head_type == "CNN_ignore":
        exit_points = [3, 6, 9, 12]

    for exit_point in exit_points:
        if exit_counts[exit_point] > 0:
            exit_accuracy[exit_point] = 100.0 * correct[exit_point] / exit_counts[exit_point]
        else:
            exit_accuracy[exit_point] = 0.0

    # Exit distribution
    exit_distribution = {k: 100.0 * v / total for k, v in exit_counts.items()}

    # Average inference time per batch
    avg_inference_time = sum(inference_times) / len(inference_times)

    results = {
        'overall_accuracy': overall_accuracy,
        'exit_accuracy': exit_accuracy,
        'exit_counts': exit_counts,
        'exit_distribution': exit_distribution,
        'avg_inference_time_per_batch': avg_inference_time,
        'predictions': all_preds,
        'labels': all_labels
    }

    print("overall_accuracy", results['overall_accuracy'], "\n",
          "exit_accuracy", results['exit_accuracy'], "\n",
          "exit_counts", results['exit_counts'], "\n",
          "exit_distribution", results['exit_distribution'], "\n",
          "avg_inference_time_per_batch", results['avg_inference_time_per_batch'], "\n"
          )

    return results


def threshold_inference_vit(model, images, thresholds):
    """
    Perform threshold-based inference for ViT models with early exits

    Args:
        model: ViT model with early exits
        images: Input images
        thresholds: List of confidence thresholds for early exits

    Returns:
        predicted_labels: Tensor of predicted class labels
        exit_points: List of exit points (0-based index)
    """
    batch_size = images.shape[0]
    exit_points = []
    predicted_labels = []

    # Get all outputs from the model
    with torch.no_grad():
        main_out, out9, out6, out3 = model(images)

    # Process each image individually
    for i in range(batch_size):
        # Check exit 3 (earliest exit)
        logits3 = out3[i].unsqueeze(0)
        softmax3 = F.softmax(logits3, dim=1)
        confidence3 = torch.max(softmax3).item()

        if confidence3 > thresholds[2]:
            # Use exit 3 (earliest exit)
            prediction = torch.argmax(logits3, dim=1)
            predicted_labels.append(prediction.item())
            exit_points.append(2)  # 0-based index
            continue

        # Check exit 6
        logits6 = out6[i].unsqueeze(0)
        softmax6 = F.softmax(logits6, dim=1)
        confidence6 = torch.max(softmax6).item()

        if confidence6 > thresholds[1]:
            # Use exit 6
            prediction = torch.argmax(logits6, dim=1)
            predicted_labels.append(prediction.item())
            exit_points.append(1)  # 0-based index
            continue

        # Check exit 9
        logits9 = out9[i].unsqueeze(0)
        softmax9 = F.softmax(logits9, dim=1)
        confidence9 = torch.max(softmax9).item()

        if confidence9 > thresholds[0]:
            # Use exit 9
            prediction = torch.argmax(logits9, dim=1)
            predicted_labels.append(prediction.item())
            exit_points.append(0)  # 0-based index
            continue

        # Use main exit (final output)
        logits_main = main_out[i].unsqueeze(0)
        prediction = torch.argmax(logits_main, dim=1)
        predicted_labels.append(prediction.item())
        exit_points.append(3)  # Main exit (0-based index)

    return torch.tensor(predicted_labels).to(images.device), exit_points


class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, annotations_file, class_to_idx, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        with open(annotations_file, 'r') as f:
            for line in f:
                img, wnid = line.strip().split('\t')[:2]
                if wnid in class_to_idx:
                    self.samples.append((img, class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.val_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def download_and_prepare_tiny_imagenet(data_dir, batch_size):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_dir, "tiny-imagenet-200")
    
    # Download if not present
    if not os.path.exists(extract_path):
        print("Downloading Tiny ImageNet...")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Done.")
    else:
        print("Tiny ImageNet already downloaded and extracted.")
    
    # Prepare data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Resize to 224x224 for ViT
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dir = os.path.join(data_dir, "tiny-imagenet-200", "train")
    val_dir = os.path.join(data_dir, "tiny-imagenet-200", "val", "images")
    val_annotations = os.path.join(data_dir, "tiny-imagenet-200", "val", "val_annotations.txt")

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = TinyImageNetValDataset(val_dir, val_annotations, class_to_idx, transform=val_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Use validation set as test set for simplicity
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_loader, val_loader, test_loader


def test_per_class_accuracy(model, test_loader, n_classes=200):
    """
    Test classification accuracy for each class in TinyImageNet.

    Args:
        model: The PyTorch model to test
        test_loader: DataLoader for the test set
        n_classes: Number of classes (default 200 for TinyImageNet)

    Returns:
        Dictionary with overall and per-class accuracy metrics
    """
    # Load model weights if path is provided
    model.eval()

    # Initialize counters
    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    # Track overall accuracy
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing per-class accuracy"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass - get final output only
            if head_type == "MLP_EE":
                outputs = model(inputs)[0]  # Get output12 (final output)
            elif head_type == "CNN_ignore":
                outputs = model(inputs)[0]  # Get output12 (final output)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Track overall accuracy
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()

            # Track per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i in range(n_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            per_class_accuracy[i] = accuracy
        else:
            per_class_accuracy[i] = 0.0

    # Calculate overall accuracy
    overall_accuracy = 100 * total_correct / total_samples

    # Sort classes by accuracy
    best_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True)[:10]
    worst_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1])[:10]

    # Get class names
    class_names = {}

    # If the dataset is TinyImageNet, get the mapping from wnids to readable names
    if dataset_type == "tinyimagenet":
        try:
            # Try to get class to index mapping from dataset
            if isinstance(test_loader.dataset, TinyImageNetValDataset):
                # For TinyImageNetValDataset, which has class_to_idx
                class_to_idx = test_loader.dataset.class_to_idx
                idx_to_class = {v: k for k, v in class_to_idx.items()}

                # Load words.txt to map wnids to human-readable names
                words_path = os.path.join("Tiny Imagenet", "tiny-imagenet-200", "words.txt")
                if os.path.exists(words_path):
                    with open(words_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                wnid, words = parts[0], parts[1]
                                human_readable = words.split(',')[0].strip()  # Take first description
                                if wnid in idx_to_class.values():
                                    for idx, cls in idx_to_class.items():
                                        if cls == wnid:
                                            class_names[idx] = f"{wnid} ({human_readable})"
                else:
                    # If words.txt not found, use wnids as class names
                    for idx, wnid in idx_to_class.items():
                        class_names[idx] = wnid
            else:
                # For regular ImageFolder dataset
                idx_to_class = {v: k for k, v in test_loader.dataset.class_to_idx.items()}
                for idx, class_dir in idx_to_class.items():
                    class_names[idx] = class_dir
        except Exception as e:
            print(f"Error getting class names: {e}")
            # Use generic class names if there's an error
            for i in range(n_classes):
                class_names[i] = f"Class {i}"

    # Print results
    print(f"Overall accuracy: {overall_accuracy:.2f}%")

    print("\nTop 10 classes by accuracy:")
    for class_idx, acc in best_classes:
        class_name = class_names.get(class_idx, f"Class {class_idx}")
        print(f"  {class_name}: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")

    print("\nBottom 10 classes by accuracy:")
    for class_idx, acc in worst_classes:
        class_name = class_names.get(class_idx, f"Class {class_idx}")
        print(f"  {class_name}: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")

    return {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'best_classes': best_classes,
        'worst_classes': worst_classes,
        'class_names': class_names
    }


def load_single_class(root_dir, class_idx, split='train', batch_size=128, shuffle=True, num_workers=4):
    if split == 'train':
        # For training set, use ImageFolder directly since structure is class-based
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Get all classes
        train_dataset = datasets.ImageFolder(
            os.path.join(root_dir, 'train'),
            transform=transform
        )

        # Get indices of samples belonging to the target class
        class_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == class_idx]

        # Create a subset dataset
        class_dataset = torch.utils.data.Subset(train_dataset, class_indices)

    elif split == 'val':
        # For validation set, we need to use our TinyImageNetValDataset
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Get class_to_idx mapping from the train set
        train_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train'))
        class_to_idx = train_dataset.class_to_idx

        # Create full validation dataset
        val_dataset = TinyImageNetValDataset(
            os.path.join(root_dir, 'val', 'images'),
            os.path.join(root_dir, 'val', 'val_annotations.txt'),
            class_to_idx,
            transform
        )

        # Get indices of samples belonging to the target class
        class_indices = [i for i, (_, label) in enumerate(val_dataset.samples) if label == class_idx]

        # Create a subset dataset
        class_dataset = torch.utils.data.Subset(val_dataset, class_indices)

    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    # Create and return the DataLoader
    dataloader = torch.utils.data.DataLoader(
        class_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    print(f"Created DataLoader for class {class_idx} ({split} split): {len(class_dataset)} samples")
    return dataloader

head_type = "CNN_ignore"

# Example usage and testing
if __name__ == "__main__":

    #seeds: 2025 8 610 128
    pl.seed_everything(2025)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset selection
    dataset_type = "tinyimagenet"  # Options: "cifar10" or "tinyimagenet"
    
    # Hyperparameters
    batch_size = 128
    num_workers = 4
    num_epochs = 100
    weight_decay = 0.005
    backbone_lr = 1e-6
    ee_lr = 1e-3
    thresholds = [0.99, 0.99, 0.90]  # acc limit
    # thresholds = [0.45, 0.45, 0.45]  # entropy limit

    SINGLE_CLASS = 0
    head_type = "CNN_ignore"  # CNN_ignore, MLP_EE
    confidence_type = "max_prob"


    # Set number of classes based on dataset
    if dataset_type == "cifar10":
        n_classes = 10
    elif dataset_type == "tinyimagenet":
        n_classes = 200
    
    # Load dataset
    if dataset_type == "cifar10":
        root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'  # Adjust path as needed
        dataprep = Data_prep_224_normal_N(root)
        train_loader, val_loader, test_loader = dataprep.create_loaders(batch_size=batch_size, num_workers=num_workers)
    elif  dataset_type == "tinyimagenet":
        if SINGLE_CLASS:
            train_loader = load_single_class(
                root_dir='Tiny Imagenet/tiny-imagenet-200',
                class_idx=10,
                split='train',
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers)

            # Create DataLoader for validation set
            val_loader = load_single_class(
                root_dir='Tiny Imagenet/tiny-imagenet-200',
                class_idx=10,
                split='val',
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)

            test_loader = val_loader
        else:
            data_dir = "Tiny Imagenet"
            train_loader, val_loader, test_loader = download_and_prepare_tiny_imagenet(data_dir, batch_size)

    # Initialize model and load pretrained weights
    pretrained_vit = models.vit_b_16()
    pretrained_vit.heads = nn.Linear(pretrained_vit.hidden_dim, n_classes)
    pretrained_vit = pretrained_vit.to(device)
    pretrained_vit.load_state_dict(torch.load('weights/vitb16_tiny_imagenet_best.pth'))

    model = create_vit_base_16_ee(n_classes=n_classes)
    model = model.to(device)
    updated_ee_vit, skipped_params = transfer_weights_from_vit_to_ee_vit(pretrained_vit, model)
    updated_ee_vit = transfer_weights(pretrained_vit, updated_ee_vit)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    backbone_params = []
    ee_params = []

    for name, param in model.named_parameters():
        if head_type == "MLP_EE":
            if 'MLP_EE' in name:
                ee_params.append(param)
            else:
                backbone_params.append(param)
        elif head_type == "CNN_ignore":
            if 'MLP_CNN' in name or 'CNN_ignore' in name:
                ee_params.append(param)
            else:
                backbone_params.append(param)

    # Include final_head in ee_params
    # for name, param in model.named_parameters():
    #     if 'final_head' in name:
    #         ee_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': ee_params, 'lr': ee_lr}
    ]

    # better use SGD when loading weights 
    # optimizer = optim.SGD(
    #     param_groups,
    #     momentum=0.9,
    #     weight_decay=weight_decay)

    # Use AdamW optimizer
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training or loading weights
    # TRAIN = True
    TRAIN = False
    if TRAIN:
        train(updated_ee_vit)
    else:
        # Load model based on dataset type and head type
        weights_path = f'weights/vit_ee_{dataset_type}_{head_type}_best.pth'
        if os.path.exists(weights_path):
            updated_ee_vit.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            log_message(f"Loaded weights from {weights_path}")
        else:
            print(f"No weights found at {weights_path}, using initialized model")
            log_message(f"No weights found at {weights_path}, using initialized model")

        # Testing
        results = validate(updated_ee_vit, test_loader, criterion)
        print(results)
        log_message(str(results))
        # Use for thresholded inference (early exiting)
        # threshold_inference(updated_ee_vit, test_loader, thresholds)

        # Run whiteboard tests if using CIFAR-10
        # if dataset_type == "cifar10":
        #     VGG16_white_board_test()
        #     Resnet50_white_board_test()

        per_class_results = test_per_class_accuracy(updated_ee_vit, test_loader, n_classes=200)


