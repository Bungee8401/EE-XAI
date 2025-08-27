import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import pickle
from tqdm import tqdm

# Try importing diffusion libraries
try:
    from diffusers import DDPMPipeline, StableDiffusionPipeline, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
    print("✓ Diffusers library loaded successfully")
except ImportError as e:
    print(f"Warning: Diffusers not available: {e}")
    DIFFUSERS_AVAILABLE = False

# Try importing torchvision with comprehensive error handling
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
    print("✓ Torchvision transforms loaded successfully")
except ImportError as e:
    print(f"Warning: Torchvision import failed: {e}")
    TORCHVISION_AVAILABLE = False
except RuntimeError as e:
    if "operator torchvision::nms does not exist" in str(e):
        print("Error: PyTorch/Torchvision version mismatch detected!")
        print("Please reinstall with: pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"Warning: Torchvision runtime error: {e}")
    TORCHVISION_AVAILABLE = False
except Exception as e:
    print(f"Warning: Unexpected torchvision error: {e}")
    TORCHVISION_AVAILABLE = False


class CIFAR100DiffusionGenerator:
    """
    CIFAR-100 style image generator using pretrained diffusion models
    """
    
    def __init__(self, device=None):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required. Install with: pip install diffusers")
            
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # CIFAR-100 complete class names organized by superclasses
        self.cifar100_fine_labels = [
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
        
        # CIFAR-100 coarse labels (superclasses)
        self.cifar100_coarse_labels = [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 
            'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
            'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]
        
        # Mapping from fine labels to coarse labels
        self.fine_to_coarse_mapping = [
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18,
            17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5,
            8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ]
        
        # For backward compatibility
        self.cifar100_classes = self.cifar100_fine_labels
        
        self.cifar_pipeline = None
        self.stable_pipeline = None
        
    def load_cifar_diffusion_model(self):
        """Load CIFAR-10 DDPM model (most similar to CIFAR-100)"""
        try:
            print("Loading CIFAR-10 DDPM model...")
            # Try with safetensors first, fallback to standard weights
            try:
                self.cifar_pipeline = DDPMPipeline.from_pretrained(
                    "google/ddpm-cifar10-32",
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
            except:
                print("Safetensors not available, using standard weights...")
                self.cifar_pipeline = DDPMPipeline.from_pretrained(
                    "google/ddpm-cifar10-32",
                    torch_dtype=torch.float32,
                    use_safetensors=False
                )
            
            self.cifar_pipeline = self.cifar_pipeline.to(self.device)
            print("✓ CIFAR diffusion model loaded successfully!")
            return True
        except Exception as e:
            print(f"✗ Failed to load CIFAR model: {e}")
            return False
    
    def load_stable_diffusion_model(self):
        """Load Stable Diffusion for text-conditioned generation"""
        try:
            print("Loading Stable Diffusion v1.5...")
            self.stable_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            self.stable_pipeline = self.stable_pipeline.to(self.device)
            self.stable_pipeline.scheduler = DDIMScheduler.from_config(self.stable_pipeline.scheduler.config)
            print("✓ Stable Diffusion model loaded successfully!")
            return True
        except Exception as e:
            print(f"✗ Failed to load Stable Diffusion: {e}")
            return False
    
    def generate_cifar_style_images(self, num_images=10, method="cifar", save_dir=None, class_name=None, class_id=None):
        """
        Generate CIFAR-100 style images
        
        Args:
            num_images: Number of images to generate
            method: "cifar" for CIFAR-10 DDPM or "stable" for Stable Diffusion
            save_dir: Directory to save images
            class_name: Specific class name to generate (for conditional generation)
            class_id: Specific class ID (0-99) to generate (for conditional generation)
        """
        if method == "cifar":
            return self._generate_with_cifar_model(num_images, save_dir, class_name, class_id)
        elif method == "stable":
            return self._generate_with_stable_diffusion(num_images, save_dir, class_name, class_id)
        else:
            # Try both methods
            images = self._generate_with_cifar_model(num_images, save_dir, class_name, class_id)
            if images is None:
                images = self._generate_with_stable_diffusion(num_images, save_dir, class_name, class_id)
            return images
    
    def generate_class_specific_images(self, class_name, num_images=10, method="stable", save_dir=None):
        """
        Generate images for a specific CIFAR-100 class
        
        Args:
            class_name: Name of the class (e.g., 'apple', 'tiger', 'airplane')
            num_images: Number of images to generate
            method: Generation method ("stable" recommended for class-specific)
            save_dir: Directory to save images
        """
        if class_name not in self.cifar100_fine_labels:
            print(f"Error: '{class_name}' not found in CIFAR-100 classes")
            print(f"Available classes: {', '.join(self.cifar100_fine_labels[:10])}...")
            return None
        
        class_id = self.cifar100_fine_labels.index(class_name)
        print(f"Generating {num_images} images for class '{class_name}' (ID: {class_id})")
        
        return self.generate_cifar_style_images(
            num_images=num_images,
            method=method,
            save_dir=save_dir,
            class_name=class_name,
            class_id=class_id
        )
    
    def _generate_with_cifar_model(self, num_images, save_dir, class_name=None, class_id=None):
        """Generate using CIFAR-10 DDPM model"""
        if self.cifar_pipeline is None:
            if not self.load_cifar_diffusion_model():
                return None
        
        generated_images = []
        used_labels = []
        
        # Note: CIFAR-10 DDPM is unconditional, so we'll generate random images
        # but label them according to the requested class if specified
        if class_name or class_id is not None:
            target_class = class_name if class_name else self.cifar100_fine_labels[class_id]
            print(f"Generating {num_images} images using CIFAR DDPM (Note: unconditional model, will label as '{target_class}')")
        else:
            print(f"Generating {num_images} images using CIFAR DDPM...")
        
        for i in tqdm(range(num_images)):
            try:
                with torch.no_grad():
                    # Generate 32x32 image
                    result = self.cifar_pipeline(
                        generator=torch.Generator(device=self.device).manual_seed(random.randint(0, 100000))
                    )
                    image_32 = result.images[0]
                    
                    # Resize for display (keeping pixelated look)
                    image_display = image_32.resize((128, 128), Image.NEAREST)
                    generated_images.append(image_display)
                    
                    # Assign label
                    if class_name or class_id is not None:
                        if class_id is not None:
                            used_labels.append(class_id)
                        else:
                            used_labels.append(self.cifar100_fine_labels.index(class_name))
                    else:
                        used_labels.append(random.randint(0, len(self.cifar100_fine_labels) - 1))
                    
            except Exception as e:
                print(f"Error generating image {i}: {e}")
                continue
        
        if save_dir and generated_images:
            method_name = f"cifar_ddpm_{class_name}" if class_name else "cifar_ddpm"
            self._save_images(generated_images, save_dir, method_name, labels=used_labels)
        
        return generated_images
    
    def _generate_with_stable_diffusion(self, num_images, save_dir, class_name=None, class_id=None):
        """Generate using Stable Diffusion with CIFAR-100 prompts"""
        if self.stable_pipeline is None:
            if not self.load_stable_diffusion_model():
                return None
        
        generated_images = []
        used_prompts = []
        used_labels = []
        
        # Determine target class
        if class_id is not None:
            target_class = self.cifar100_fine_labels[class_id]
            target_id = class_id
        elif class_name:
            target_class = class_name
            target_id = self.cifar100_fine_labels.index(class_name)
        else:
            target_class = None
            target_id = None
        
        print(f"Generating {num_images} images using Stable Diffusion" + 
              (f" for class '{target_class}'" if target_class else ""))
        
        for i in tqdm(range(num_images)):
            try:
                # Create class-specific or random prompt
                if target_class:
                    class_name_for_prompt = target_class
                    current_label = target_id
                else:
                    current_label = random.randint(0, len(self.cifar100_fine_labels) - 1)
                    class_name_for_prompt = self.cifar100_fine_labels[current_label]
                
                # Create enhanced prompt based on class type
                prompt = self._create_enhanced_prompt(class_name_for_prompt)
                
                with torch.no_grad():
                    # Generate high-res image
                    result = self.stable_pipeline(
                        prompt,
                        height=512,
                        width=512,
                        num_inference_steps=25,  # Increased for better quality
                        guidance_scale=8.0,      # Slightly higher for better class adherence
                        generator=torch.Generator(device=self.device).manual_seed(random.randint(0, 100000))
                    )
                    image_512 = result.images[0]
                    
                    # Convert to CIFAR-100 style (32x32 then upscale)
                    image_32 = image_512.resize((32, 32), Image.LANCZOS)
                    image_display = image_32.resize((128, 128), Image.NEAREST)
                    
                    generated_images.append(image_display)
                    used_prompts.append(f"{class_name_for_prompt}: {prompt}")
                    used_labels.append(current_label)
                    
            except Exception as e:
                print(f"Error generating image {i}: {e}")
                continue
        
        if save_dir and generated_images:
            method_name = f"stable_diffusion_{target_class}" if target_class else "stable_diffusion"
            self._save_images(generated_images, save_dir, method_name, used_prompts, used_labels)
        
        return generated_images
    
    def _create_enhanced_prompt(self, class_name):
        """Create enhanced prompts for better CIFAR-100 style generation"""
        # Enhanced prompts for different categories
        prompt_templates = {
            # Animals
            'bear': "a simple pixel art of a brown bear, 32x32 resolution, minimal details, centered, solid background",
            'tiger': "a simple pixel art of an orange tiger with black stripes, 32x32 resolution, minimal, centered",
            'elephant': "a simple pixel art of a gray elephant, 32x32 resolution, side view, minimal details, solid background",
            'whale': "a simple pixel art of a blue whale, 32x32 resolution, swimming, minimal, centered, solid background",
            'dolphin': "a simple pixel art of a gray dolphin, 32x32 resolution, jumping, minimal details, blue background",
            'shark': "a simple pixel art of a gray shark, 32x32 resolution, side view, minimal, blue background",
            'lion': "a simple pixel art of a golden lion, 32x32 resolution, minimal mane, centered, solid background",
            'leopard': "a simple pixel art of a spotted leopard, 32x32 resolution, minimal details, centered",
            
            # Vehicles
            'bicycle': "a simple pixel art of a red bicycle, 32x32 resolution, side view, minimal details, centered",
            'motorcycle': "a simple pixel art of a black motorcycle, 32x32 resolution, side view, minimal, centered",
            'pickup_truck': "a simple pixel art of a pickup truck, 32x32 resolution, side view, minimal details, centered",
            'train': "a simple pixel art of a train locomotive, 32x32 resolution, side view, minimal, centered",
            'tractor': "a simple pixel art of a green tractor, 32x32 resolution, side view, minimal details, centered",
            'tank': "a simple pixel art of a military tank, 32x32 resolution, side view, minimal, olive green",
            'rocket': "a simple pixel art of a white rocket, 32x32 resolution, pointing up, minimal details, blue background",
            
            # Household items
            'chair': "a simple pixel art of a wooden chair, 32x32 resolution, front view, minimal details, centered",
            'table': "a simple pixel art of a brown table, 32x32 resolution, minimal details, centered, solid background",
            'bed': "a simple pixel art of a bed with pillow, 32x32 resolution, side view, minimal, centered",
            'couch': "a simple pixel art of a sofa, 32x32 resolution, front view, minimal details, centered",
            'lamp': "a simple pixel art of a table lamp, 32x32 resolution, minimal details, centered, solid background",
            'telephone': "a simple pixel art of an old telephone, 32x32 resolution, minimal details, centered",
            'television': "a simple pixel art of a TV screen, 32x32 resolution, front view, minimal, centered",
            
            # Food and plants
            'apple': "a simple pixel art of a red apple, 32x32 resolution, minimal details, centered, white background",
            'orange': "a simple pixel art of an orange fruit, 32x32 resolution, minimal details, centered, white background",
            'pear': "a simple pixel art of a green pear, 32x32 resolution, minimal details, centered, white background",
            'mushroom': "a simple pixel art of a red mushroom with white spots, 32x32 resolution, minimal, centered",
            'rose': "a simple pixel art of a red rose flower, 32x32 resolution, minimal details, green stem, centered",
            'tulip': "a simple pixel art of a red tulip flower, 32x32 resolution, minimal details, green stem, centered",
            'sunflower': "a simple pixel art of a yellow sunflower, 32x32 resolution, minimal details, centered",
            
            # Nature
            'mountain': "a simple pixel art of a snow-capped mountain, 32x32 resolution, minimal details, blue sky background",
            'forest': "a simple pixel art of green trees, 32x32 resolution, minimal details, forest scene, centered",
            'cloud': "a simple pixel art of white fluffy clouds, 32x32 resolution, minimal, blue sky background",
            'sea': "a simple pixel art of blue ocean waves, 32x32 resolution, minimal details, blue water",
            
            # Buildings
            'house': "a simple pixel art of a small house, 32x32 resolution, front view, minimal details, centered",
            'castle': "a simple pixel art of a medieval castle, 32x32 resolution, minimal details, centered, stone gray",
            'skyscraper': "a simple pixel art of a tall building, 32x32 resolution, minimal details, centered, blue sky",
        }
        
        # Use specific prompt if available, otherwise create generic one
        if class_name in prompt_templates:
            return prompt_templates[class_name]
        else:
            return f"a simple pixel art of a {class_name.replace('_', ' ')}, 32x32 resolution, minimal details, centered, solid background"
    
    def _save_images(self, images, save_dir, method_name, prompts=None, labels=None):
        """Save generated images to directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual images
        for i, image in enumerate(images):
            filename = f"{method_name}_{i:04d}.png"
            image.save(os.path.join(save_dir, filename))
        
        # Create and save grid
        self._create_image_grid(images, save_dir, f"{method_name}_grid.png")
        
        # Save prompts if available
        if prompts:
            with open(os.path.join(save_dir, f"{method_name}_prompts.txt"), 'w') as f:
                for i, prompt in enumerate(prompts):
                    f.write(f"{i:04d}: {prompt}\n")
        
        # Save labels if available
        if labels:
            with open(os.path.join(save_dir, f"{method_name}_labels.txt"), 'w') as f:
                for i, label in enumerate(labels):
                    class_name = self.cifar100_fine_labels[label]
                    f.write(f"{i:04d}: {label} ({class_name})\n")
        
        print(f"✓ Saved {len(images)} images to {save_dir}")
        if labels:
            unique_classes = set(labels)
            print(f"  - Generated {len(unique_classes)} different classes")
            if len(unique_classes) <= 5:
                class_names = [self.cifar100_fine_labels[l] for l in unique_classes]
                print(f"  - Classes: {', '.join(class_names)}")
    
    def _create_image_grid(self, images, save_dir, filename):
        """Create a grid of images"""
        if not images:
            return
        
        n_images = len(images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        # Get image size
        img_size = images[0].size[0]
        
        # Create grid
        grid_img = Image.new('RGB', (grid_size * img_size, grid_size * img_size), 'white')
        
        for i, img in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            x = col * img_size
            y = row * img_size
            grid_img.paste(img, (x, y))
        
        grid_img.save(os.path.join(save_dir, filename))
    
    def create_dataset(self, num_images=1000, save_dir="./data_split/diffusion_cifar100", method="stable", balanced=True):
        """Create a synthetic CIFAR-100 style dataset
        
        Args:
            num_images: Total number of images to generate
            save_dir: Directory to save the dataset
            method: Generation method ("stable" for class-conditional, "cifar" for unconditional)
            balanced: If True, generate equal number of images per class
        """
        print(f"Creating dataset with {num_images} images...")
        
        if balanced and method == "stable":
            # Generate balanced dataset with equal samples per class
            images_per_class = max(1, num_images // len(self.cifar100_fine_labels))
            total_images = images_per_class * len(self.cifar100_fine_labels)
            print(f"Generating balanced dataset: {images_per_class} images per class ({total_images} total)")
            
            all_images = []
            all_labels = []
            
            for class_id, class_name in enumerate(self.cifar100_fine_labels):
                print(f"\nGenerating class {class_id+1}/{len(self.cifar100_fine_labels)}: {class_name}")
                
                class_images = self.generate_class_specific_images(
                    class_name=class_name,
                    num_images=images_per_class,
                    method=method,
                    save_dir=None  # Don't save individual class images
                )
                
                if class_images:
                    all_images.extend(class_images)
                    all_labels.extend([class_id] * len(class_images))
                
                # Progress update
                if (class_id + 1) % 10 == 0:
                    print(f"Completed {class_id + 1}/{len(self.cifar100_fine_labels)} classes")
            
            images = all_images
            labels = all_labels
            
        else:
            # Generate random dataset
            images = self.generate_cifar_style_images(num_images, method=method, save_dir=save_dir)
            
            if not images:
                print("Failed to generate images for dataset")
                return None
            
            # Assign labels
            if method == "stable":
                # Labels should be assigned during generation for stable diffusion
                labels = [random.randint(0, len(self.cifar100_fine_labels) - 1) for _ in range(len(images))]
            else:
                labels = [random.randint(0, len(self.cifar100_fine_labels) - 1) for _ in range(len(images))]
        
        if not images:
            print("Failed to generate images for dataset")
            return None
        
        # Convert to dataset format
        dataset_arrays = []
        
        for i, img in enumerate(images):
            # Convert to 32x32 numpy array
            img_32 = img.resize((32, 32), Image.LANCZOS)
            img_array = np.array(img_32, dtype=np.uint8)
            dataset_arrays.append(img_array)
        
        # Create dataset dictionary
        dataset = {
            'data': np.array(dataset_arrays),
            'labels': np.array(labels),
            'fine_labels': labels,  # CIFAR-100 format
            'coarse_labels': [self.fine_to_coarse_mapping[label] for label in labels],
            'fine_label_names': self.cifar100_fine_labels,
            'coarse_label_names': self.cifar100_coarse_labels,
            'filenames': [f"diffusion_{i:06d}.png" for i in range(len(images))],
            'metadata': {
                'num_samples': len(images),
                'image_shape': (32, 32, 3),
                'num_fine_classes': len(self.cifar100_fine_labels),
                'num_coarse_classes': len(self.cifar100_coarse_labels),
                'generation_method': method,
                'balanced': balanced
            }
        }
        
        # Save dataset
        os.makedirs(save_dir, exist_ok=True)
        dataset_path = os.path.join(save_dir, f"diffusion_cifar100_{method}_dataset.pkl")
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save individual images if not saved during generation
        if balanced and method == "stable":
            images_dir = os.path.join(save_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            for i, (img, label) in enumerate(zip(images, labels)):
                class_name = self.cifar100_fine_labels[label]
                filename = f"{class_name}_{i:06d}.png"
                img_32 = img.resize((32, 32), Image.LANCZOS)
                img_32.save(os.path.join(images_dir, filename))
        
        print(f"✓ Dataset saved to {dataset_path}")
        print(f"  - {dataset['metadata']['num_samples']} images")
        print(f"  - {dataset['metadata']['num_fine_classes']} fine classes")
        print(f"  - {dataset['metadata']['num_coarse_classes']} coarse classes")
        print(f"  - Image shape: {dataset['metadata']['image_shape']}")
        print(f"  - Method: {dataset['metadata']['generation_method']}")
        print(f"  - Balanced: {dataset['metadata']['balanced']}")
        
        # Print class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"  - Class distribution: {len(unique_labels)} classes represented")
        if len(unique_labels) <= 10:
            for label, count in zip(unique_labels, counts):
                class_name = self.cifar100_fine_labels[label]
                print(f"    {class_name}: {count} images")
        
        return dataset
    
    def demo(self, num_images=8, demo_classes=None):
        """Run a demonstration of the diffusion generator
        
        Args:
            num_images: Total number of images to generate
            demo_classes: List of specific classes to demo (e.g., ['apple', 'tiger', 'bicycle'])
        """
        print("CIFAR-100 Diffusion Generator Demo")
        print("=" * 50)
        
        # Create results directory
        demo_dir = "./Results/Diffusion_Demo"
        os.makedirs(demo_dir, exist_ok=True)
        
        all_images = []
        methods = []
        
        if demo_classes:
            # Generate specific classes
            print(f"\nGenerating class-specific images for: {', '.join(demo_classes)}")
            images_per_class = max(1, num_images // len(demo_classes))
            
            for class_name in demo_classes:
                print(f"\n2. Generating {images_per_class} images for class '{class_name}'...")
                class_images = self.generate_class_specific_images(
                    class_name=class_name,
                    num_images=images_per_class,
                    method="stable",
                    save_dir=demo_dir
                )
                
                if class_images:
                    all_images.extend(class_images)
                    methods.extend([f"Class: {class_name}"] * len(class_images))
        else:
            # Try CIFAR model first
            print("\n1. Trying CIFAR-10 DDPM model...")
            images_cifar = self._generate_with_cifar_model(num_images//2, demo_dir)
            
            # Try Stable Diffusion with random classes
            print("\n2. Trying Stable Diffusion with random classes...")
            images_stable = self._generate_with_stable_diffusion(num_images//2, demo_dir)
            
            if images_cifar:
                all_images.extend(images_cifar)
                methods.extend(["CIFAR DDPM"] * len(images_cifar))
            
            if images_stable:
                all_images.extend(images_stable)
                methods.extend(["Stable Diffusion"] * len(images_stable))
        
        if all_images:
            self._create_comparison_plot(all_images, methods, demo_dir)
            print(f"\n✓ Demo completed! Results saved to {demo_dir}")
            
            # Demo class-specific generation
            if not demo_classes:
                print("\n3. Demonstrating class-specific generation...")
                sample_classes = ['apple', 'tiger', 'bicycle', 'house']
                for class_name in sample_classes[:2]:  # Generate 2 classes for demo
                    print(f"Generating 2 images of '{class_name}'...")
                    class_images = self.generate_class_specific_images(
                        class_name=class_name,
                        num_images=2,
                        method="stable",
                        save_dir=demo_dir
                    )
                    if class_images:
                        print(f"✓ Generated {len(class_images)} images of {class_name}")
        else:
            print("\n✗ Demo failed - no images generated")
        
        return all_images
    
    def list_available_classes(self):
        """List all available CIFAR-100 classes"""
        print("Available CIFAR-100 classes:")
        print("=" * 30)
        
        for i, (coarse_name, fine_labels) in enumerate(zip(self.cifar100_coarse_labels, 
                                                           [[] for _ in range(len(self.cifar100_coarse_labels))])):
            # Group fine labels by coarse labels
            fine_in_coarse = [j for j, coarse_id in enumerate(self.fine_to_coarse_mapping) if coarse_id == i]
            fine_names = [self.cifar100_fine_labels[j] for j in fine_in_coarse]
            
            print(f"\n{coarse_name.replace('_', ' ').title()}:")
            for name in fine_names:
                print(f"  - {name}")
        
        print(f"\nTotal: {len(self.cifar100_fine_labels)} fine classes in {len(self.cifar100_coarse_labels)} coarse categories")
    
    def _create_comparison_plot(self, images, methods, save_dir):
        """Create a comparison plot"""
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        fig.suptitle('Generated CIFAR-100 Style Images', fontsize=16)
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(n_images):
            axes[i].imshow(images[i])
            axes[i].set_title(f'{methods[i]}\nImage {i+1}', fontsize=10)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main demonstration function"""
    print("CIFAR-100 Diffusion Image Generator")
    print("=" * 50)
    
    try:
        # Create generator
        generator = CIFAR100DiffusionGenerator()
        
        # List available classes
        print("\nFirst, let's see what classes are available:")
        generator.list_available_classes()
        
        # Demo class-specific generation
        print("\n" + "=" * 50)
        print("DEMO 1: Class-specific generation")
        demo_classes = ['apple', 'tiger', 'bicycle', 'house']
        images = generator.demo(num_images=8, demo_classes=demo_classes)
        
        if images:
            print(f"\n✓ Successfully generated {len(images)} class-specific images!")
            
            # Demo single class generation
            print("\n" + "=" * 50)
            print("DEMO 2: Single class generation")
            
            single_class_images = generator.generate_class_specific_images(
                class_name='tiger',
                num_images=4,
                method='stable',
                save_dir='./Results/Diffusion_Demo/single_class'
            )
            
            if single_class_images:
                print(f"✓ Generated {len(single_class_images)} tiger images!")
            
            # Optionally create a small balanced dataset
            print("\n" + "=" * 50)
            print("DEMO 3: Creating balanced dataset")
            print("Creating small balanced dataset (5 images per class for first 10 classes)...")
            
            # Create dataset with subset of classes for demo
            dataset = generator.create_dataset(
                num_images=50,  # 5 images per class for 10 classes
                method="stable",
                balanced=True,
                save_dir="./data_split/demo_diffusion_cifar100"
            )
            
            if dataset:
                print("✓ Dataset creation successful!")
                print(f"Dataset contains {dataset['metadata']['num_samples']} images")
                print(f"Balanced: {dataset['metadata']['balanced']}")
            else:
                print("✗ Dataset creation failed")
        else:
            print("\n✗ Image generation failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# Additional utility functions for easy usage
def generate_single_class(class_name, num_images=10, save_dir="./generated_images"):
    """Convenience function to generate images for a single class"""
    generator = CIFAR100DiffusionGenerator()
    return generator.generate_class_specific_images(
        class_name=class_name,
        num_images=num_images,
        method="stable",
        save_dir=save_dir
    )

def create_balanced_cifar100_dataset(num_images_per_class=10, save_dir="./data_split/balanced_diffusion_cifar100"):
    """Convenience function to create a balanced CIFAR-100 dataset"""
    generator = CIFAR100DiffusionGenerator()
    total_images = num_images_per_class * 100  # 100 classes in CIFAR-100
    return generator.create_dataset(
        num_images=total_images,
        method="stable",
        balanced=True,
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
