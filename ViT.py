import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from CustomDataset import Data_prep_224_normal_N


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, C, H, W] -> [B, E, H//P, W//P] -> [B, E, N] -> [B, N, E]
        x = self.proj(x)  # [B, E, H//P, W//P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, E]
        B, N, E = x.shape

        # Linear projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # each with shape [B, num_heads, N, head_dim]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0., attn_dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )

    def forward(self, x):
        # Layer normalization and attention with residual connection
        x = x + self.attn(self.norm1(x))
        # Layer normalization and MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Apply Xavier initialization to linear layers
        self.apply(self._init_weights_xavier)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Classification from [CLS] token
        x = x[:, 0]
        x = self.head(x)

        return x


def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    if scheduler:
        scheduler.step()

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

    return running_loss / len(dataloader), 100. * correct / total


def load_torchvision_weights():
    # Create model
    model = VisionTransformer(**vit_config)

    # Load pretrained weights from torchvision
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        print("Loading pretrained ViT weights from torchvision...")

        # Get pretrained torchvision model
        pretrained_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Transfer weights for shared architecture parts
        model_dict = model.state_dict()
        pretrained_dict = {}

        # Map corresponding layers (modify these mappings based on your model structure)
        mapping = {
            'patch_embed.proj': 'conv_proj',
            'pos_embed': 'encoder.pos_embedding',
            'cls_token': 'class_token',
            'blocks': 'encoder.layers',
            'norm': 'encoder.ln'
        }

        for pretrained_key, pretrained_value in pretrained_model.state_dict().items():
            for model_prefix, pretrained_prefix in mapping.items():
                if pretrained_key.startswith(pretrained_prefix):
                    model_key = pretrained_key.replace(pretrained_prefix, model_prefix)
                    if model_key in model_dict and model_dict[model_key].shape == pretrained_value.shape:
                        pretrained_dict[model_key] = pretrained_value

        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        # Reinitialize the classification head
        nn.init.xavier_uniform_(model.head.weight)
        if model.head.bias is not None:
            nn.init.constant_(model.head.bias, 0)

    except (ImportError, RuntimeError) as e:
        print(f"Could not load pretrained weights: {e}")
        print("Proceeding with randomly initialized weights")

    return model


def get_pretrained_vit():
    # Import the pretrained ViT model
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    # Load pretrained model with weights
    print("Loading pretrained ViT model from torchvision...")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    print(model)

    # Modify the classifier head for CIFAR-10 (10 classes)
    model.heads[-1] = nn.Linear(model.hidden_dim, 10)

    # Initialize the new classification head
    nn.init.xavier_uniform_(model.heads[-1].weight)
    if model.heads[-1].bias is not None:
        nn.init.constant_(model.heads[-1].bias, 0)

    print("Pretrained ViT model loaded and modified for CIFAR-10")
    return model


if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 100
    num_workers = 2
    num_epochs = 20
    learning_rate = 3e-4
    weight_decay = 0.05

    # Model configuration
    vit_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_channels': 3,
        'num_classes': 10,
        'embed_dim': 384,  # Smaller than original ViT for CIFAR-10
        'depth': 6,  # Fewer layers
        'num_heads': 6,  # Fewer attention heads
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'attn_dropout': 0.0
    }

    # Load dataset
    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'  # Adjust path as needed
    dataprep = Data_prep_224_normal_N(root)
    train_loader, val_loader, test_loader = dataprep.create_loaders(batch_size=batch_size,num_workers=num_workers)

    # Initialize model
    model = get_pretrained_vit()
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )



    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights/vit_cifar10_best.pth')
            print(f"Saved best model with acc: {best_val_acc:.2f}%")

    # Load best model for testing
    model.load_state_dict(torch.load('weights/vit_cifar10_best.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print(model)

