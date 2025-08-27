"""
CIFAR-100 Diffusion Model Training Script
Supports both training from scratch and fine-tuning existing models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
import argparse
from tqdm import tqdm
import math
from typing import Optional, Union, List, Tuple
import json
from datetime import datetime
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline, StableDiffusionPipeline
from diffusers.training_utils import EMAModel
DIFFUSERS_AVAILABLE = True
print("✓ Diffusers library loaded successfully")



class CIFAR100DiffusionDataset(Dataset):
    """Custom dataset for CIFAR-100 diffusion training"""
    
    def __init__(self, root_dir='./CIFAR100', train=True, transform=None, class_idx=None):
        """
        Args:
            root_dir: Root directory for CIFAR-100 data
            train: If True, use training data, else test data
            transform: Transform to apply to images
            class_idx: If specified, only load data for this class (0-99)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.class_idx = class_idx
        
        # Load CIFAR-100 dataset
        self.cifar100 = datasets.CIFAR100(
            root=root_dir, 
            train=train, 
            download=True, 
            transform=None  # We'll apply transforms manually
        )
        
        # Filter by class if specified
        if class_idx is not None:
            if not (0 <= class_idx <= 99):
                raise ValueError("class_idx must be between 0 and 99")
            
            # Find indices for the specified class
            class_indices = [i for i, target in enumerate(self.cifar100.targets) if target == class_idx]
            self.indices = class_indices
            print(f"Loaded {len(self.indices)} samples for class {class_idx}")
        else:
            self.indices = list(range(len(self.cifar100)))
            print(f"Loaded {len(self.indices)} samples for all classes")
            
        # CIFAR-100 class names
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.cifar100[real_idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = transforms.ToTensor()(image)
        
        return image, label


class CIFAR100ConditionalDiffusion:
    def __init__(self, 
                 image_size: int = 224,  # Changed default to 224
                 num_classes: int = 100,
                 device: str = 'auto',
                 num_train_timesteps: int = 1000,  # Increased for larger images
                 class_conditional: bool = True):
        """
        Args:
            image_size: Size of images (224 for upscaled CIFAR-100)
            num_classes: Number of classes (100 for CIFAR-100)
            device: Device to use ('auto', 'cuda', 'cpu')
            num_train_timesteps: Number of diffusion timesteps
            class_conditional: Whether to use class conditioning
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_train_timesteps = num_train_timesteps
        self.class_conditional = class_conditional
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model and scheduler
        self.unet = None
        self.scheduler = None
        self.ema_model = None
        self.optimizer = None
        
        # Training metrics
        self.training_history = []
        
    def initialize_model(self, pretrained_model: Optional[str] = None):
        """
        Initialize or load the UNet model
        
        Args:
            pretrained_model: Path to pretrained model or HuggingFace model name
        """
        if pretrained_model and pretrained_model.startswith('google/ddpm-cifar10'):
            # Load pretrained CIFAR-10 model and adapt for CIFAR-100
            print(f"Loading pretrained model: {pretrained_model}")
            pipeline = DDPMPipeline.from_pretrained(pretrained_model)
            self.unet = pipeline.unet
            self.scheduler = pipeline.scheduler
            
            # Modify for class conditioning if needed
            if self.class_conditional and self.unet.class_embed_type != "identity":
                # Add class embedding
                self.unet.num_class_embeds = self.num_classes
                print(f"Added class conditioning for {self.num_classes} classes")
            
        elif pretrained_model and os.path.exists(pretrained_model):
            # Load from local file
            print(f"Loading model from: {pretrained_model}")
            checkpoint = torch.load(pretrained_model, map_location=self.device)
            
            # Load config if available
            config = checkpoint.get('config', {})
            
            # Create model architecture first, then load weights
            # This ensures compatibility with different model architectures
            if 'model' in checkpoint:
                # This is our custom checkpoint format
                self.unet = None  # Will be created below
                self.scheduler = checkpoint.get('scheduler', None)
                # Store the state dict to load after model creation
                model_state_dict = checkpoint['model']
            else:
                # This might be a different checkpoint format
                raise ValueError(f"Unsupported checkpoint format in {pretrained_model}")
            
            # Create the model architecture
            if self.unet is None:
                print("Creating UNet architecture...")
                if self.image_size == 224:
                    # Architecture optimized for 224x224 images
                    self.unet = UNet2DModel(
                        sample_size=self.image_size,
                        in_channels=3,
                        out_channels=3,
                        layers_per_block=2,
                        block_out_channels=(64, 128, 256, 512, 768, 1024),
                        down_block_types=(
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "AttnDownBlock2D",
                            "AttnDownBlock2D",
                            "DownBlock2D",
                        ),
                        up_block_types=(
                            "UpBlock2D",
                            "AttnUpBlock2D",
                            "AttnUpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                        ),
                        num_class_embeds=self.num_classes if self.class_conditional else None,
                        attention_head_dim=8,
                        norm_num_groups=32,
                        dropout=0.1,
                    )
                else:
                    # Original architecture for 32x32 images
                    self.unet = UNet2DModel(
                        sample_size=self.image_size,
                        in_channels=3,
                        out_channels=3,
                        layers_per_block=2,
                        block_out_channels=(128, 128, 256, 256, 512, 512),
                        down_block_types=(
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "AttnDownBlock2D",
                            "DownBlock2D",
                        ),
                        up_block_types=(
                            "UpBlock2D",
                            "AttnUpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                        ),
                        num_class_embeds=self.num_classes if self.class_conditional else None,
                    )
            
            # Load the state dict
            self.unet.load_state_dict(model_state_dict)
            print("✓ Model weights loaded successfully")
            
        else:
            # Create new model from scratch - optimized for 224x224 images
            print(f"Creating new UNet model from scratch for {self.image_size}x{self.image_size} images")
            
            if self.image_size == 224:
                # Architecture optimized for 224x224 images
                self.unet = UNet2DModel(
                    sample_size=self.image_size,
                    in_channels=3,
                    out_channels=3,
                    layers_per_block=2,
                    block_out_channels=(64, 128, 256, 512, 768, 1024),  # More channels for larger images
                    down_block_types=(
                        "DownBlock2D",      # 224 -> 112
                        "DownBlock2D",      # 112 -> 56  
                        "DownBlock2D",      # 56 -> 28
                        "AttnDownBlock2D",  # 28 -> 14 (attention at reasonable resolution)
                        "AttnDownBlock2D",  # 14 -> 7
                        "DownBlock2D",      # 7 -> 3-4
                    ),
                    up_block_types=(
                        "UpBlock2D",        # 3-4 -> 7
                        "AttnUpBlock2D",    # 7 -> 14
                        "AttnUpBlock2D",    # 14 -> 28
                        "UpBlock2D",        # 28 -> 56
                        "UpBlock2D",        # 56 -> 112
                        "UpBlock2D",        # 112 -> 224
                    ),
                    num_class_embeds=self.num_classes if self.class_conditional else None,
                    attention_head_dim=8,
                    norm_num_groups=32,
                    dropout=0.1,
                )
            else:
                # Original architecture for 32x32 images
                self.unet = UNet2DModel(
                    sample_size=self.image_size,
                    in_channels=3,
                    out_channels=3,
                    layers_per_block=2,
                    block_out_channels=(128, 128, 256, 256, 512, 512),
                    down_block_types=(
                        "DownBlock2D",
                        "DownBlock2D",
                        "DownBlock2D",
                        "DownBlock2D",
                        "AttnDownBlock2D",
                        "DownBlock2D",
                    ),
                    up_block_types=(
                        "UpBlock2D",
                        "AttnUpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                    ),
                    num_class_embeds=self.num_classes if self.class_conditional else None,
                )
        
        # Initialize scheduler if not loaded
        if self.scheduler is None:
            # Use DDPMScheduler for training with scaled_linear (widely supported)
            self.scheduler = DDPMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule="scaled_linear",  # More widely supported than cosine
                prediction_type="epsilon",
                variance_type="fixed_small_log",  # Better for training stability
                clip_sample=False,  # Don't clip for better gradients
                beta_start=0.0001,  # Good defaults for larger images
                beta_end=0.02,
            )
            
            # Also create DDIM scheduler for faster inference during generation
            self.ddim_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule="scaled_linear",  # DDIM supports scaled_linear
                prediction_type="epsilon",
                clip_sample=False,
                beta_start=0.0001,
                beta_end=0.02,
            )
        
        # Move to device
        self.unet = self.unet.to(self.device)
        
        # Initialize EMA
        self.ema_model = EMAModel(
            parameters=self.unet.parameters(),
            power=0.75,
            decay=0.9999,
            min_decay=0.0001,
        )
        
        print(f"✓ Model initialized with {sum(p.numel() for p in self.unet.parameters()):,} parameters")
        
    def setup_optimizer(self, learning_rate: float = 1e-4, weight_decay: float = 0.01):
        """Setup optimizer with learning rate scheduling for 224x224 images"""
        self.optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Add learning rate scheduler for better training with larger images
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            total_steps = len(self.train_loader) * 100  # Assume 100 epochs max
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=min(500, total_steps // 10),
                num_training_steps=total_steps
            )
            print(f"✓ Optimizer and LR scheduler setup with lr={learning_rate}")
        else:
            self.lr_scheduler = None
            print(f"✓ Optimizer setup with lr={learning_rate} (LR scheduler will be added after data loaders)")
        
    def create_data_loaders(self, 
                          root_dir: str = './CIFAR100',
                          batch_size: int = 32,
                          num_workers: int = 4,
                          class_idx: Optional[int] = None,
                          val_split: float = 0.1):
        """
        Create training and validation data loaders
        
        Args:
            root_dir: Root directory for CIFAR-100 data
            batch_size: Batch size
            num_workers: Number of data loading workers
            class_idx: If specified, only train on this class
            val_split: Fraction of training data to use for validation
        """
        # Define transforms for 224x224 images (no resize needed since we set to 224)
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize CIFAR-100 from 32x32 to 224x224
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Add slight rotation for better augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize CIFAR-100 from 32x32 to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Create datasets
        full_train_dataset = CIFAR100DiffusionDataset(
            root_dir=root_dir,
            train=True,
            transform=train_transform,
            class_idx=class_idx
        )
        
        test_dataset = CIFAR100DiffusionDataset(
            root_dir=root_dir,
            train=False,
            transform=val_transform,
            class_idx=class_idx
        )
        
        # Split training data for validation
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✓ Data loaders created:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        if class_idx is not None:
            class_name = full_train_dataset.class_names[class_idx]
            print(f"  Training on class {class_idx}: {class_name}")
        
        # Setup learning rate scheduler now that we have train_loader
        if hasattr(self, 'optimizer') and self.optimizer is not None and not hasattr(self, 'lr_scheduler'):
            total_steps = len(self.train_loader) * 100  # Assume 100 epochs max
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=min(500, total_steps // 10),
                num_training_steps=total_steps
            )
            print(f"✓ Learning rate scheduler added with {total_steps} total steps")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def train_step(self, batch, epoch: int, step: int):
        """Single training step"""
        images, labels = batch
        images = images.to(self.device)
        
        if self.class_conditional:
            labels = labels.to(self.device)
        else:
            labels = None
        
        # Sample noise and timesteps
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (images.shape[0],), device=self.device
        ).long()
        
        # Add noise to images
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        
        # Predict noise
        if self.class_conditional:
            noise_pred = self.unet(noisy_images, timesteps, class_labels=labels).sample
        else:
            noise_pred = self.unet(noisy_images, timesteps).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update learning rate scheduler if available
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Update EMA
        self.ema_model.step(self.unet.parameters())
        
        return loss.item()
    
    def validate(self):
        """Validation step"""
        self.unet.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch
                images = images.to(self.device)
                
                if self.class_conditional:
                    labels = labels.to(self.device)
                else:
                    labels = None
                
                # Sample noise and timesteps
                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (images.shape[0],), device=self.device
                ).long()
                
                # Add noise to images
                noisy_images = self.scheduler.add_noise(images, noise, timesteps)
                
                # Predict noise
                if self.class_conditional:
                    noise_pred = self.unet(noisy_images, timesteps, class_labels=labels).sample
                else:
                    noise_pred = self.unet(noisy_images, timesteps).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
                num_batches += 1
        
        self.unet.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def generate_samples(self, 
                        num_samples: int = 16, 
                        class_labels: Optional[List[int]] = None,
                        use_ema: bool = True,
                        use_ddim: bool = True,
                        num_inference_steps: int = 50):
        """Generate samples from the trained model"""
        # Use EMA weights for generation if available
        if use_ema and self.ema_model is not None:
            self.ema_model.store(self.unet.parameters())
            self.ema_model.copy_to(self.unet.parameters())
        
        self.unet.eval()
        
        # Choose scheduler for generation
        if use_ddim and hasattr(self, 'ddim_scheduler'):
            generation_scheduler = self.ddim_scheduler
            generation_scheduler.set_timesteps(num_inference_steps)
            timesteps = generation_scheduler.timesteps
        else:
            generation_scheduler = self.scheduler
            timesteps = generation_scheduler.timesteps
        
        with torch.no_grad():
            # Initialize random noise
            shape = (num_samples, 3, self.image_size, self.image_size)
            noise = torch.randn(shape, device=self.device)
            
            # Prepare class labels
            if self.class_conditional:
                if class_labels is None:
                    # Generate random class labels
                    class_labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)
                else:
                    class_labels = torch.tensor(class_labels, device=self.device)
            
            # Denoising loop
            desc = f"Generating ({'DDIM' if use_ddim else 'DDPM'})"
            for t in tqdm(timesteps, desc=desc):
                timesteps_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                
                if self.class_conditional:
                    noise_pred = self.unet(noise, timesteps_batch, class_labels=class_labels).sample
                else:
                    noise_pred = self.unet(noise, timesteps_batch).sample
                
                # Update noise
                noise = generation_scheduler.step(noise_pred, t, noise).prev_sample
        
        # Restore original weights
        if use_ema and self.ema_model is not None:
            self.ema_model.restore(self.unet.parameters())
        
        self.unet.train()
        
        # Denormalize and convert to PIL images
        images = (noise / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype(np.uint8)
        
        pil_images = [Image.fromarray(img) for img in images]
        
        return pil_images, class_labels.cpu().numpy() if self.class_conditional else None
    
    def train(self, 
              num_epochs: int = 100,
              save_every: int = 20,
              generate_every: int = 5,
              save_dir: str = './training_weights/diffusion_cifar100',
              log_every: int = 100):
        """
        Main training loop
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save model every N epochs
            generate_every: Generate samples every N epochs
            save_dir: Directory to save models and samples
            log_every: Log metrics every N steps
        """
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model will be saved every {save_every} epochs to {save_dir}")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.unet.train()
            epoch_loss = 0
            num_batches = 0
            
            # Training loop
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch, epoch, step)
                epoch_loss += loss
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Log metrics
                if global_step % log_every == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Step {global_step}, Avg Loss: {avg_loss:.4f}")
            
            # Validation
            val_loss = self.validate()
            avg_train_loss = epoch_loss / num_batches
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save training history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'global_step': global_step
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, 'best_model.pth'), epoch, global_step)
                print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            
            # Generate samples
            if (epoch + 1) % generate_every == 0:
                print(f"Generating samples at epoch {epoch+1}...")
                try:
                    if self.class_conditional:
                        # Generate samples for a few different classes
                        class_labels = list(range(min(16, self.num_classes)))
                        if len(class_labels) < 16:
                            class_labels = class_labels * (16 // len(class_labels) + 1)
                            class_labels = class_labels[:16]
                    else:
                        class_labels = None
                    
                    samples, labels = self.generate_samples(16, class_labels)
                    self.save_samples(samples, labels, epoch+1, save_dir)
                    print("✓ Samples generated and saved")
                except Exception as e:
                    print(f"✗ Failed to generate samples: {e}")
            
            # Save model checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'), epoch, global_step)
                print(f"✓ Model checkpoint saved at epoch {epoch+1}")
        
        # Final save
        self.save_model(os.path.join(save_dir, 'final_model.pth'), num_epochs-1, global_step)
        self.save_training_history(save_dir)
        print("✓ Training completed!")
    
    def save_model(self, path: str, epoch: int, step: int):
        """Save model checkpoint"""
        checkpoint = {
            'model': self.unet.state_dict(),
            'ema_model': self.ema_model.state_dict() if self.ema_model else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler,
            'epoch': epoch,
            'step': step,
            'training_history': self.training_history,
            'config': {
                'image_size': self.image_size,
                'num_classes': self.num_classes,
                'class_conditional': self.class_conditional,
                'num_train_timesteps': self.num_train_timesteps
            }
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        config = checkpoint.get('config', {})
        self.image_size = config.get('image_size', 32)
        self.num_classes = config.get('num_classes', 100)
        self.class_conditional = config.get('class_conditional', True)
        self.num_train_timesteps = config.get('num_train_timesteps', 1000)
        
        # Initialize model if not done
        if self.unet is None:
            self.initialize_model()
        
        # Load states
        self.unet.load_state_dict(checkpoint['model'])
        if checkpoint.get('ema_model') and self.ema_model:
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        if checkpoint.get('optimizer') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler'):
            self.scheduler = checkpoint['scheduler']
        
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"✓ Model loaded from {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Step: {checkpoint.get('step', 'unknown')}")
    
    def save_samples(self, samples: List[Image.Image], labels: Optional[np.ndarray], epoch: int, save_dir: str):
        """Save generated samples"""
        # Create grid
        grid_size = int(np.ceil(np.sqrt(len(samples))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, (sample, ax) in enumerate(zip(samples, axes)):
            ax.imshow(sample)
            ax.axis('off')
            if labels is not None and i < len(labels):
                dataset = CIFAR100DiffusionDataset()
                class_name = dataset.class_names[labels[i]]
                ax.set_title(f'{class_name}', fontsize=8)
        
        # Hide empty subplots
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'samples', f'epoch_{epoch}_samples.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual samples
        samples_dir = os.path.join(save_dir, 'samples', f'epoch_{epoch}')
        os.makedirs(samples_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            label_name = dataset.class_names[labels[i]] if labels is not None and i < len(labels) else f'sample_{i}'
            sample.save(os.path.join(samples_dir, f'{label_name}_{i:03d}.png'))
    
    def save_training_history(self, save_dir: str):
        """Save training history"""
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Plot training curves
        if self.training_history:
            epochs = [h['epoch'] for h in self.training_history]
            train_losses = [h['train_loss'] for h in self.training_history]
            val_losses = [h['val_loss'] for h in self.training_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, label='Training Loss', alpha=0.8)
            plt.plot(epochs, val_losses, label='Validation Loss', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CIFAR-100 Diffusion Model')
    parser.add_argument('--mode', choices=['train', 'finetune', 'generate'], default='finetune', help='Mode: train from scratch, finetune existing model, or generate samples')
    parser.add_argument('--pretrained', type=str, default=None, help=' google/ddpm-cifar10-32')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--class_idx', type=int, default=0, help='Train on specific class only (0-99)')
    parser.add_argument('--save_dir', type=str, default='./training_weights/diffusion_cifar100', help='Directory to save models')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--conditional', action='store_true', default=True, help='Use class conditioning')
    parser.add_argument('--unconditional', action='store_false', dest='conditional', help='Train unconditional model')
    
    # Generation options
    parser.add_argument('--generate_num', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--generate_class', type=int, default=None, help='Specific class to generate')
    
    args = parser.parse_args()
    
    if not DIFFUSERS_AVAILABLE:
        print("Error: diffusers library is required")
        print("Install with: pip install diffusers accelerate")
        return
    
    # Create trainer
    trainer = CIFAR100ConditionalDiffusion(
        image_size=32,  # Use 32 for CIFAR-10/100 compatibility
        num_classes=100,
        device=args.device,
        class_conditional=args.conditional
    )
    
    if args.mode == 'generate':
        # Generate samples only
        if args.pretrained is None:
            print("Error: --pretrained model path required for generation")
            return
        
        print("Loading model for generation...")
        trainer.load_model(args.pretrained)
        
        print(f"Generating {args.generate_num} samples...")
        if args.generate_class is not None:
            class_labels = [args.generate_class] * args.generate_num
        else:
            class_labels = None
        
        samples, labels = trainer.generate_samples(args.generate_num, class_labels)
        
        # Save samples
        os.makedirs('./generated_samples', exist_ok=True)
        trainer.save_samples(samples, labels, 0, './generated_samples')
        print("✓ Samples saved to ./generated_samples")
        
    else:
        # Training or fine-tuning
        print(f"Mode: {args.mode}")
        
        # Initialize model
        if args.mode == 'finetune' and args.pretrained:
            trainer.initialize_model(args.pretrained)
        else:
            trainer.initialize_model()
        
        # Setup optimizer
        trainer.setup_optimizer(args.lr)
        
        # Create data loaders
        trainer.create_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_idx=args.class_idx
        )
        
        # Train
        trainer.train(
            num_epochs=args.epochs,
            save_dir=args.save_dir
        )


if __name__ == '__main__':
    main()
