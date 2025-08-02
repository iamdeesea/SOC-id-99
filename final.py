# ============================================================================
# FACE AGING SYSTEM WITH CAAE - IMPORTS FIRST
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import subprocess
import zipfile
import shutil

# ============================================================================
# KAGGLE DATASET DOWNLOADER CLASS
# ============================================================================

class KaggleDatasetDownloader:
    """Handle Kaggle dataset downloading"""

    def __init__(self, kaggle_json_path):
        self.kaggle_json_path = kaggle_json_path
        self.setup_kaggle_api()

    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        # Create .kaggle directory in home folder
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)

        # Copy kaggle.json to the correct location
        target_path = os.path.join(kaggle_dir, "kaggle.json")
        if not os.path.exists(target_path):
            shutil.copy2(self.kaggle_json_path, target_path)
            os.chmod(target_path, 0o600)  # Set correct permissions

        print(f"✅ Kaggle API credentials setup complete")

    def download_dataset(self, dataset_name, download_path):
        """Download dataset from Kaggle"""
        try:
            print(f"📥 Downloading dataset: {dataset_name}")

            # Use kaggle API command
            cmd = [
                "kaggle", "datasets", "download",
                "-d", dataset_name,
                "-p", download_path,
                "--unzip"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Dataset downloaded successfully to: {download_path}")
                return True
            else:
                print(f"❌ Download failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False

    def list_dataset_files(self, dataset_name):
        """List files in a Kaggle dataset"""
        try:
            cmd = ["kaggle", "datasets", "files", dataset_name]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("📋 Dataset files:")
                print(result.stdout)
                return result.stdout.split('\n')
            else:
                print(f"❌ Failed to list files: {result.stderr}")
                return []

        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []

class KaggleDatasetLoader:
    """Enhanced loader for Kaggle Human Faces dataset with auto-download"""

    def __init__(self, dataset_path, auto_download=False, kaggle_json_path=None):
        self.dataset_path = dataset_path
        self.auto_download = auto_download
        self.kaggle_json_path = kaggle_json_path
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        if auto_download and kaggle_json_path:
            self.downloader = KaggleDatasetDownloader(kaggle_json_path)

    def load_dataset(self):
        """Load the Kaggle human faces dataset with auto-download option"""

        # Try auto-download if enabled and dataset doesn't exist
        if self.auto_download and not os.path.exists(self.dataset_path):
            print(f"📁 Dataset path not found: {self.dataset_path}")
            print("📥 Attempting to download from Kaggle...")

            # Create parent directory
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

            # Download dataset
            success = self.downloader.download_dataset(
                'ashwingupta3012/human-faces',
                os.path.dirname(self.dataset_path)
            )

            if not success:
                print("❌ Auto-download failed. Please manually download the dataset.")
                return [], []

        # Now load the dataset
        image_paths = []
        ages = []

        print(f"📂 Loading dataset from: {self.dataset_path}")

        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path not found: {self.dataset_path}")
            print("💡 Please download the dataset from: https://www.kaggle.com/datasets/ashwingupta3012/human-faces")
            return [], []

        # Method 1: Try to find CSV file with age information
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]

        if csv_files:
            print(f"📄 Found CSV file: {csv_files[0]}")
            csv_path = os.path.join(self.dataset_path, csv_files[0])
            try:
                df = pd.read_csv(csv_path)
                print(f"📊 CSV columns: {df.columns.tolist()}")

                # Common column names for images and ages
                image_col = None
                age_col = None

                for col in df.columns:
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['image', 'file', 'path', 'name']):
                        image_col = col
                    if any(word in col_lower for word in ['age', 'years', 'old']):
                        age_col = col

                if image_col and age_col:
                    for _, row in df.iterrows():
                        img_name = row[image_col]
                        age = row[age_col]

                        # Find the actual image file
                        img_path = self._find_image_file(img_name)
                        if img_path:
                            image_paths.append(img_path)
                            ages.append(int(age))

                    print(f"✅ Loaded {len(image_paths)} images with ages from CSV")
                    return image_paths, ages

            except Exception as e:
                print(f"⚠️ Error reading CSV: {e}")

        # Method 2: Scan directories and assign pseudo-ages based on folder structure
        print("🔍 No CSV found or failed to parse. Scanning directories...")
        image_paths, ages = self._scan_directories()

        if not image_paths:
            # Method 3: Flat directory structure - assign random ages
            print("🔍 Scanning flat directory structure...")
            image_paths, ages = self._scan_flat_directory()

        return image_paths, ages

    def _find_image_file(self, filename):
        """Find image file in dataset directory"""
        # Try exact match first
        for root, dirs, files in os.walk(self.dataset_path):
            if filename in files:
                return os.path.join(root, filename)

        # Try without extension
        base_name = os.path.splitext(filename)[0]
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if os.path.splitext(file)[0] == base_name:
                    return os.path.join(root, file)

        return None

    def _scan_directories(self):
        """Scan directory structure for age-based folders"""
        image_paths = []
        ages = []

        for root, dirs, files in os.walk(self.dataset_path):
            # Check if directory name contains age information
            dir_name = os.path.basename(root).lower()

            # Try to extract age from directory name
            age = self._extract_age_from_name(dir_name)

            if age is not None:
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.image_extensions):
                        full_path = os.path.join(root, file)
                        image_paths.append(full_path)
                        ages.append(age)

        print(f"📁 Found {len(image_paths)} images in age-organized directories")
        return image_paths, ages

    def _scan_flat_directory(self):
        """Scan flat directory and assign pseudo-ages"""
        image_paths = []
        ages = []

        # Get all image files
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)

                    # Try to extract age from filename
                    age = self._extract_age_from_name(file)
                    if age is None:
                        # Assign pseudo-random age based on filename hash
                        import hashlib
                        hash_val = int(hashlib.md5(file.encode()).hexdigest(), 16)
                        age = 20 + (hash_val % 50)  # Ages between 20-70

                    ages.append(age)

        print(f"📁 Found {len(image_paths)} images in flat directory structure")
        return image_paths, ages

    def _extract_age_from_name(self, name):
        """Extract age from filename or directory name"""
        import re

        # Look for age patterns
        age_patterns = [
            r'age[_\-]?(\d+)',
            r'(\d+)[_\-]?years?',
            r'(\d+)[_\-]?yo',
            r'(\d+)[_\-]?old',
            r'_(\d+)_',
            r'^(\d+)',
        ]

        for pattern in age_patterns:
            match = re.search(pattern, name.lower())
            if match:
                age = int(match.group(1))
                if 1 <= age <= 100:  # Reasonable age range
                    return age

        # Check for age group keywords
        age_keywords = {
            'child': 8, 'kid': 8, 'baby': 3, 'infant': 1,
            'teen': 16, 'teenager': 16, 'adolescent': 15,
            'young': 22, 'youth': 20,
            'adult': 35, 'middle': 45,
            'old': 60, 'senior': 65, 'elderly': 70
        }

        for keyword, age in age_keywords.items():
            if keyword in name.lower():
                return age

        return None

# ============================================================================
# 1. IMAGE PREPROCESSING PIPELINE
# ============================================================================

class FacePreprocessor:
    """Handles face detection, alignment, and preprocessing"""

    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def detect_face(self, image):
        """Detect and crop face from image"""
        if isinstance(image, str):
            image = cv2.imread(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        # Take the largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Add padding
        padding = int(0.2 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        face = image[y:y+h, x:x+w]
        return face

    def preprocess_image(self, image_path):
        """Complete preprocessing pipeline"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    return None
            else:
                image = image_path

            # Detect and crop face
            face = self.detect_face(image)
            if face is None:
                return None

            # Convert to PIL and apply transforms
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            tensor = self.transform(pil_image)

            return tensor
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

# ============================================================================
# 2. DATASET CLASS FOR PYTORCH DATALOADERS
# ============================================================================

class FaceAgingDataset(Dataset):
    """Dataset class for face aging with age labels"""

    def __init__(self, image_paths, ages, preprocessor, augment=False):
        self.image_paths = image_paths
        self.ages = ages
        self.preprocessor = preprocessor
        self.augment = augment

        # Age groups mapping
        self.age_groups = {
            0: (0, 12),    # Child
            1: (13, 25),   # Young Adult
            2: (26, 40),   # Adult
            3: (41, 60),   # Middle Age
            4: (61, 100)   # Senior
        }

        # Augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10),
            ])

    def age_to_group(self, age):
        """Convert numerical age to age group"""
        for group, (min_age, max_age) in self.age_groups.items():
            if min_age <= age <= max_age:
                return group
        return 2  # Default to adult if age is unclear

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        age = self.ages[idx]

        # Preprocess image
        tensor = self.preprocessor.preprocess_image(image_path)

        if tensor is None:
            # Return a dummy tensor if preprocessing fails
            tensor = torch.zeros(3, 128, 128)

        # Apply augmentation if enabled
        if self.augment and hasattr(self, 'aug_transform'):
            # Convert back to PIL for augmentation, then back to tensor
            pil_img = transforms.ToPILImage()(tensor * 0.5 + 0.5)  # Denormalize
            pil_img = self.aug_transform(pil_img)
            tensor = transforms.ToTensor()(pil_img)
            tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)

        age_group = self.age_to_group(age)

        return {
            'image': tensor,
            'age': torch.tensor(age, dtype=torch.float32),
            'age_group': torch.tensor(age_group, dtype=torch.long),
            'path': image_path
        }

# ============================================================================
# 3. CAAE MODEL ARCHITECTURE
# ============================================================================

class Encoder(nn.Module):
    """Encoder network for CAAE"""

    def __init__(self, input_dim=3, latent_dim=100):
        super(Encoder, self).__init__()

        self.conv_layers = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 64 x 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 32 x 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 x 16 x 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 512 x 8 x 8
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)  # Added dropout
        )

        # 1024 x 4 x 4
        self.fc = nn.Linear(1024 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    """Decoder network for CAAE"""

    def __init__(self, latent_dim=100, age_dim=5, output_dim=3):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.age_dim = age_dim

        self.fc = nn.Linear(latent_dim + age_dim, 1024 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            # 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Added dropout

            # 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 64 x 64
            nn.ConvTranspose2d(64, output_dim, 4, 2, 1),
            nn.Tanh()
            # Output: 3 x 128 x 128
        )

    def forward(self, z, age_label):
        # Concatenate latent code with age label
        if len(age_label.shape) == 1:
            age_label = F.one_hot(age_label, num_classes=self.age_dim).float()

        combined = torch.cat([z, age_label], dim=1)
        x = self.fc(combined)
        x = x.view(x.size(0), 1024, 4, 4)
        x = self.deconv_layers(x)
        return x

class Discriminator(nn.Module):
    """Discriminator for adversarial training"""

    def __init__(self, input_dim=3):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Added dropout

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv_layers(x).view(-1, 1).squeeze(1)

class CAAE(nn.Module):
    """Complete CAAE model"""

    def __init__(self, latent_dim=100, age_dim=5):
        super(CAAE, self).__init__()

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, age_dim=age_dim)
        self.discriminator = Discriminator()

        self.latent_dim = latent_dim
        self.age_dim = age_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, age_label):
        return self.decoder(z, age_label)

    def forward(self, x, age_label):
        z = self.encode(x)
        reconstructed = self.decode(z, age_label)
        return z, reconstructed

# ============================================================================
# 4. TRAINING AND INFERENCE PIPELINE
# ============================================================================

class FaceAgingPipeline:
    """Complete pipeline for face aging"""

    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.preprocessor = FacePreprocessor()

        # Initialize models
        self.model = CAAE().to(device)

        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Initialize optimizers
        self.optimizer_AE = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=0.0002, betas=(0.5, 0.999)
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def create_dataloader(self, image_paths, ages, batch_size=32, shuffle=True, augment=False):
        """Create DataLoader for training/testing"""
        dataset = FaceAgingDataset(image_paths, ages, self.preprocessor, augment=augment)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss_AE = 0
        total_loss_D = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            age_groups = batch['age_group'].to(self.device)

            batch_size = images.size(0)

            # Train Discriminator
            self.optimizer_D.zero_grad()

            # Real images
            real_labels = torch.ones(batch_size).to(self.device)
            fake_labels = torch.zeros(batch_size).to(self.device)

            real_output = self.model.discriminator(images)
            d_loss_real = self.bce_loss(real_output, real_labels)

            # Fake images
            z, fake_images = self.model(images, age_groups)
            fake_output = self.model.discriminator(fake_images.detach())
            d_loss_fake = self.bce_loss(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_D.step()

            # Train Autoencoder
            self.optimizer_AE.zero_grad()

            # Reconstruction loss
            recon_loss = self.mse_loss(fake_images, images)

            # Adversarial loss
            fake_output = self.model.discriminator(fake_images)
            adv_loss = self.bce_loss(fake_output, real_labels)

            # Total autoencoder loss
            ae_loss = recon_loss + 0.1 * adv_loss
            ae_loss.backward()
            self.optimizer_AE.step()

            total_loss_AE += ae_loss.item()
            total_loss_D += d_loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, AE Loss: {ae_loss.item():.4f}, D Loss: {d_loss.item():.4f}')

        return total_loss_AE / len(dataloader), total_loss_D / len(dataloader)

    def age_transform(self, image_path, target_age_group, save_path=None):
        """Transform image to target age group"""
        self.model.eval()

        with torch.no_grad():
            # Preprocess image
            tensor = self.preprocessor.preprocess_image(image_path)
            if tensor is None:
                print("Failed to preprocess image")
                return None

            tensor = tensor.unsqueeze(0).to(self.device)
            target_age = torch.tensor([target_age_group]).to(self.device)

            # Generate aged image
            z = self.model.encode(tensor)
            aged_image = self.model.decode(z, target_age)

            # Convert back to PIL image
            aged_image = aged_image.cpu().squeeze(0)
            aged_image = (aged_image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            aged_image = transforms.ToPILImage()(aged_image)

            if save_path:
                aged_image.save(save_path)

            return aged_image

    def save_model(self, path):
        """Save model state"""
        torch.save({
            'encoder': self.model.encoder.state_dict(),
            'decoder': self.model.decoder.state_dict(),
            'discriminator': self.model.discriminator.state_dict(),
            'optimizer_AE': self.optimizer_AE.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.decoder.load_state_dict(checkpoint['decoder'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator'])

# ============================================================================
# 5. TESTING AND EVALUATION FUNCTIONS
# ============================================================================

def test_different_conditions(pipeline, test_images, output_dir='test_results'):
    """Test model on different lighting conditions and ages"""
    os.makedirs(output_dir, exist_ok=True)

    age_groups = ['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior']

    for img_path in test_images:
        base_name = os.path.basename(img_path).split('.')[0]

        # Test all age groups
        for age_idx, age_name in enumerate(age_groups):
            output_path = os.path.join(output_dir, f'{base_name}_{age_name}.jpg')
            aged_image = pipeline.age_transform(img_path, age_idx, output_path)

            if aged_image:
                print(f"Generated {age_name} version of {base_name}")

def create_age_interpolation(pipeline, image_path, output_dir='interpolation'):
    """Create smooth age interpolation"""
    os.makedirs(output_dir, exist_ok=True)

    pipeline.model.eval()
    with torch.no_grad():
        tensor = pipeline.preprocessor.preprocess_image(image_path)
        if tensor is None:
            return

        tensor = tensor.unsqueeze(0).to(pipeline.device)
        z = pipeline.model.encode(tensor)

        # Create interpolation between age groups
        for i in range(5):  # 5 age groups
            for j in range(3):  # 3 interpolation steps per group
                alpha = j / 3.0

                # Interpolate between current and next age group
                if i < 4:
                    age1 = F.one_hot(torch.tensor([i]), num_classes=5).float().to(pipeline.device)
                    age2 = F.one_hot(torch.tensor([i+1]), num_classes=5).float().to(pipeline.device)
                    interpolated_age = (1 - alpha) * age1 + alpha * age2
                else:
                    interpolated_age = F.one_hot(torch.tensor([i]), num_classes=5).float().to(pipeline.device)

                aged_image = pipeline.model.decode(z, interpolated_age)
                aged_image = aged_image.cpu().squeeze(0)
                aged_image = (aged_image + 1) / 2
                aged_image = transforms.ToPILImage()(aged_image)

                output_path = os.path.join(output_dir, f'age_{i}_{j}.jpg')
                aged_image.save(output_path)

# ============================================================================
# 6. ENHANCED MAIN FUNCTION WITH DATASET LOADING
# ============================================================================

def initialize_kaggle_dataset(dataset_path=None, kaggle_json_path=None, auto_download=True,
                             train_split=0.8, batch_size=16):
    """Initialize the Kaggle dataset with automatic download option"""

    if dataset_path is None:
        dataset_path = "./datasets/human-faces"

    # Load dataset with auto-download option
    loader = KaggleDatasetLoader(
        dataset_path,
        auto_download=auto_download,
        kaggle_json_path=kaggle_json_path
    )

    image_paths, ages = loader.load_dataset()

    if not image_paths:
        print("❌ No images found! Please check your dataset path or kaggle.json file.")
        return None, None, None

    print(f"✅ Total images loaded: {len(image_paths)}")
    print(f"📊 Age range: {min(ages)} - {max(ages)}")

    # Create train/validation split
    train_paths, val_paths, train_ages, val_ages = train_test_split(
        image_paths, ages, train_size=train_split, random_state=42, stratify=None
    )

    print(f"🚂 Training images: {len(train_paths)}")
    print(f"✅ Validation images: {len(val_paths)}")

    # Initialize pipeline
    pipeline = FaceAgingPipeline()

    # Create dataloaders
    train_loader = pipeline.create_dataloader(
        train_paths, train_ages, batch_size=batch_size, shuffle=True, augment=True
    )

    val_loader = pipeline.create_dataloader(
        val_paths, val_ages, batch_size=batch_size, shuffle=False, augment=False
    )

    return pipeline, train_loader, val_loader

def train_model(pipeline, train_loader, val_loader, epochs=50, save_interval=10):
    """Train the face aging model"""

    print("Starting training...")
    best_loss = float('inf')
    losses_ae = []
    losses_d = []

    for epoch in range(epochs):
        # Training
        train_ae_loss, train_d_loss = pipeline.train_epoch(train_loader, epoch)
        losses_ae.append(train_ae_loss)
        losses_d.append(train_d_loss)

        # Validation
        pipeline.model.eval()
        val_ae_loss = 0
        val_d_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(pipeline.device)
                age_groups = batch['age_group'].to(pipeline.device)

                # Forward pass
                z, reconstructed = pipeline.model(images, age_groups)
                recon_loss = pipeline.mse_loss(reconstructed, images)

                val_ae_loss += recon_loss.item()

        val_ae_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train - AE Loss: {train_ae_loss:.4f}, D Loss: {train_d_loss:.4f}')
        print(f'  Val   - AE Loss: {val_ae_loss:.4f}')

        # Save best model
        if val_ae_loss < best_loss:
            best_loss = val_ae_loss
            pipeline.save_model('best_caae_model.pth')
            print(f'  New best model saved! Loss: {best_loss:.4f}')

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            pipeline.save_model(f'caae_epoch_{epoch+1}.pth')
            print(f'  Checkpoint saved: caae_epoch_{epoch+1}.pth')

        print('-' * 50)

    return losses_ae, losses_d

def main():
    """Main function with Kaggle API integration"""

    # ========================================
    # CONFIGURATION - UPDATE THESE PATHS!
    # ========================================

    KAGGLE_JSON_PATH = "/content/kaggle.json"     # 🔑 UPDATE THIS!
    DATASET_PATH = "./datasets/human-faces"        # Downloaded dataset location
    AUTO_DOWNLOAD = True                           # Set to True for automatic download
    EPOCHS = 50
    BATCH_SIZE = 16

    print("🚀 === Face Aging System with CAAE ===")
    print(f"🔑 Kaggle JSON: {KAGGLE_JSON_PATH}")
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"📥 Auto download: {AUTO_DOWNLOAD}")

    # ========================================
    # STEP 1: DOWNLOAD AND LOAD DATASET
    # ========================================

    if AUTO_DOWNLOAD:
        print("\n📥 === DOWNLOADING DATASET ===")

        # Check if kaggle.json exists
        if not os.path.exists(KAGGLE_JSON_PATH):
            print(f"❌ Kaggle JSON not found at: {KAGGLE_JSON_PATH}")
            print("💡 Please:")
            print("   1. Go to kaggle.com → Account → API → Create New API Token")
            print("   2. Download kaggle.json file")
            print("   3. Update KAGGLE_JSON_PATH in the code")
            return

        # Create downloader and download dataset
        try:
            downloader = KaggleDatasetDownloader(KAGGLE_JSON_PATH)
            print("📋 Checking dataset contents...")
            files = downloader.list_dataset_files('ashwingupta3012/human-faces')
        except Exception as e:
            print(f"❌ Error with Kaggle API: {e}")
            print("Continuing with manual dataset loading...")
            AUTO_DOWNLOAD = False

    # Initialize dataset and pipeline
    print("\n🔄 === INITIALIZING DATASET ===")
    pipeline, train_loader, val_loader = initialize_kaggle_dataset(
        dataset_path=DATASET_PATH,
        kaggle_json_path=KAGGLE_JSON_PATH,
        auto_download=AUTO_DOWNLOAD,
        batch_size=BATCH_SIZE
    )

    if pipeline is None:
        print("❌ Failed to initialize dataset.")
        print("🔧 Troubleshooting:")
        print("   1. Check your kaggle.json file is valid")
        print("   2. Ensure you've accepted the dataset terms on Kaggle")
        print("   3. Verify your internet connection")
        print("\n💡 Creating a demo with synthetic data instead...")

        # Create demo pipeline with synthetic data
        pipeline = create_demo_pipeline()
        if pipeline:
            print("✅ Demo pipeline created successfully!")
        else:
            return
    else:
        # ========================================
        # STEP 2: TRAIN MODEL
        # ========================================

        print(f"\n🚂 === TRAINING MODEL ({EPOCHS} epochs) ===")
        losses_ae, losses_d = train_model(pipeline, train_loader, val_loader, epochs=EPOCHS)

        # Create training visualization
        create_training_visualization(losses_ae, losses_d)

        # ========================================
        # STEP 3: TEST TRAINED MODEL
        # ========================================

        print("\n🧪 === TESTING TRAINED MODEL ===")

        # Load best model
        if os.path.exists('best_caae_model.pth'):
            pipeline.load_model('best_caae_model.pth')
            print("✅ Loaded best trained model")
        else:
            print("⚠️ No saved model found, using current model state")

        # Get some test images
        test_images = []
        for batch in val_loader:
            test_images.extend(batch['path'][:3])  # Take first 3 images
            break

        if test_images:
            print(f"🖼️ Testing with {len(test_images)} images...")

            # Test different age transformations
            print("🔄 Generating age transformations...")
            test_different_conditions(pipeline, test_images)

            # Create age interpolation for first test image
            print("📈 Creating age interpolation...")
            create_age_interpolation(pipeline, test_images[0])

            print("✅ Results saved to:")
            print("   📁 ./test_results/ - Age transformations")
            print("   📁 ./interpolation/ - Age interpolation")

    # ========================================
    # STEP 4: INTERACTIVE DEMO
    # ========================================

    print("\n🎮 === INTERACTIVE DEMO ===")
    print("Age groups: 0=Child, 1=Young Adult, 2=Adult, 3=Middle Age, 4=Senior")

    # Interactive demo function
    def run_interactive_demo():
        while True:
            print("\n" + "="*50)
            print("🎯 FACE AGING DEMO")
            print("="*50)

            demo_image_path = input("📸 Enter path to test image (or 'quit' to exit): ").strip()

            if demo_image_path.lower() in ['quit', 'exit', 'q']:
                break

            if not demo_image_path:
                print("⚠️ Please enter a valid image path")
                continue

            if not os.path.exists(demo_image_path):
                print(f"❌ File not found: {demo_image_path}")
                continue

            try:
                print("\n🎯 Age Group Options:")
                print("   0 = Child (0-12)")
                print("   1 = Young Adult (13-25)")
                print("   2 = Adult (26-40)")
                print("   3 = Middle Age (41-60)")
                print("   4 = Senior (61+)")

                target_age = input("🎯 Enter target age group (0-4): ").strip()

                if not target_age.isdigit() or not (0 <= int(target_age) <= 4):
                    print("❌ Please enter a number between 0-4")
                    continue

                target_age = int(target_age)
                age_names = ["Child", "Young Adult", "Adult", "Middle Age", "Senior"]

                print(f"\n🔄 Transforming to {age_names[target_age]}...")

                # Generate output filename
                base_name = os.path.splitext(os.path.basename(demo_image_path))[0]
                output_path = f"demo_result_{base_name}_{age_names[target_age].replace(' ', '_')}.jpg"

                # Transform image
                result = pipeline.age_transform(demo_image_path, target_age, output_path)

                if result:
                    print(f"✅ Success! Result saved as: {output_path}")

                    # Generate all age groups for comparison
                    comparison_choice = input("🎨 Generate all age groups for comparison? (y/n): ").strip().lower()
                    if comparison_choice in ['y', 'yes']:
                        print("🔄 Generating all age transformations...")
                        test_different_conditions(pipeline, [demo_image_path], output_dir=f"comparison_{base_name}")
                        print(f"✅ All age versions saved in: comparison_{base_name}/")

                        # Create smooth interpolation
                        interp_choice = input("📈 Create smooth age interpolation? (y/n): ").strip().lower()
                        if interp_choice in ['y', 'yes']:
                            print("📈 Creating age interpolation...")
                            create_age_interpolation(pipeline, demo_image_path, output_dir=f"interpolation_{base_name}")
                            print(f"✅ Age interpolation saved in: interpolation_{base_name}/")
                else:
                    print("❌ Failed to process image. Please check if it contains a clear face.")

            except Exception as e:
                print(f"❌ Error processing image: {e}")

    # Option to run interactive demo
    demo_choice = input("\n🎮 Run interactive demo? (y/n): ").strip().lower()
    if demo_choice in ['y', 'yes']:
        run_interactive_demo()

    print("\n🎉 === COMPLETE! ===")
    print("📋 Summary:")
    print("   ✅ System initialized successfully")
    if 'train_loader' in locals():
        print("   ✅ Dataset loaded and processed")
        print("   ✅ Model trained and saved")
        print("   ✅ Test results generated")
    print("   ✅ Interactive demo available")

    print("\n🔧 Files created:")
    if os.path.exists('best_caae_model.pth'):
        print("   📄 best_caae_model.pth - Best trained model")
    if os.path.exists('test_results'):
        print("   📁 test_results/ - Age transformation examples")
    if os.path.exists('interpolation'):
        print("   📁 interpolation/ - Smooth age transitions")

    print("\n💡 Next steps:")
    print("   1. Experiment with your own images")
    print("   2. Adjust model architecture in the code")
    print("   3. Try different training parameters")
    print("   4. Fine-tune on domain-specific data")

def create_demo_pipeline():
    """Create a demo pipeline with synthetic data for testing"""
    print("🎭 Creating demo mode with synthetic data...")

    try:
        # Initialize pipeline
        pipeline = FaceAgingPipeline()

        # Create some dummy data for basic testing
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 128, 128).to(pipeline.device)
        dummy_ages = torch.randint(0, 5, (batch_size,)).to(pipeline.device)

        # Test forward pass
        with torch.no_grad():
            z, reconstructed = pipeline.model(dummy_images, dummy_ages)
            print(f"✅ Model forward pass successful: {reconstructed.shape}")

        print("✅ Demo pipeline ready - you can test image transformations!")
        return pipeline

    except Exception as e:
        print(f"❌ Failed to create demo pipeline: {e}")
        return None

# ============================================================================
# 9. ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def quick_setup(kaggle_json_path):
    """Quick setup function for first-time users"""

    print("🚀 QUICK SETUP - Face Aging System")
    print("="*50)

    # Check requirements
    requirements_ok = True
    missing_packages = []

    # Check Python packages
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy'
    }

    print("📦 Checking required packages...")
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - Not installed")
            missing_packages.append(package_name)
            requirements_ok = False

    if not requirements_ok:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)} kaggle")

        # Try to auto-install (optional)
        auto_install = input("\n🤖 Auto-install missing packages? (y/n): ").strip().lower()
        if auto_install in ['y', 'yes']:
            print("📦 Installing packages...")
            try:
                import subprocess
                import sys
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install'
                ] + missing_packages + ['kaggle'],
                capture_output=True, text=True)

                if result.returncode == 0:
                    print("✅ Packages installed successfully!")
                    requirements_ok = True
                else:
                    print(f"❌ Installation failed: {result.stderr}")
            except Exception as e:
                print(f"❌ Auto-installation failed: {e}")

        if not requirements_ok:
            return False

    # Check kaggle.json (only if path is provided and not default)
    if kaggle_json_path != "path/to/your/kaggle.json":
        if not os.path.exists(kaggle_json_path):
            print(f"\n❌ Kaggle JSON not found: {kaggle_json_path}")
            print("📝 To get your kaggle.json:")
            print("   1. Go to https://www.kaggle.com/")
            print("   2. Login → Account → API → 'Create New API Token'")
            print("   3. Download kaggle.json")
            print("   4. Update the path in the code")
            return False
        else:
            print(f"   ✅ kaggle.json found")
    else:
        print("\n⚠️ Using default kaggle.json path - please update KAGGLE_JSON_PATH")

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ GPU available: {torch.cuda.get_device_name()}")
        else:
            print("   ⚠️ No GPU detected - training will be slower on CPU")
    except:
        print("   ⚠️ Cannot check GPU availability")

    print("\n✅ Setup complete! Ready to run main()")
    return True

def benchmark_model(pipeline, test_loader, num_samples=100):
    """Benchmark model performance"""

    print(f"🔬 Benchmarking model on {num_samples} samples...")

    pipeline.model.eval()
    total_time = 0
    successful_transforms = 0

    import time

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx * test_loader.batch_size >= num_samples:
                break

            images = batch['image'].to(pipeline.device)
            age_groups = batch['age_group'].to(pipeline.device)

            # Time the forward pass
            start_time = time.time()
            z, reconstructed = pipeline.model(images, age_groups)
            end_time = time.time()

            total_time += (end_time - start_time)
            successful_transforms += images.size(0)

    avg_time = total_time / successful_transforms
    fps = 1.0 / avg_time

    print(f"📊 Benchmark Results:")
    print(f"   ⏱️ Average time per image: {avg_time*1000:.2f}ms")
    print(f"   🚀 Throughput: {fps:.1f} FPS")
    print(f"   ✅ Successful transforms: {successful_transforms}/{num_samples}")

    return avg_time, fps

def create_training_visualization(losses_ae, losses_d, save_path="training_curves.png"):
    """Create training visualization"""

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses_ae, label='Autoencoder Loss', color='blue')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses_d, label='Discriminator Loss', color='red')
    plt.title('Discriminator Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Training curves saved: {save_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration - UPDATE THIS PATH!
    KAGGLE_JSON_PATH = "/content/kaggle.json"  # 🔑 UPDATE THIS!

    print("🔍 Running quick setup check...")

    # Run setup check
    setup_ok = quick_setup(KAGGLE_JSON_PATH)

    if setup_ok:
        try:
            main()
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            print("\n🔧 Troubleshooting:")
            print("   1. Ensure all packages are installed correctly")
            print("   2. Check your kaggle.json file path and permissions")
            print("   3. Verify you have sufficient disk space")
            print("   4. Check your internet connection for downloads")

            import traceback
            print(f"\n🐛 Full error details:")
            traceback.print_exc()
    else:
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        print("\n💡 Quick fix commands:")
        print("pip install pillow scikit-learn")
        print("# Then update KAGGLE_JSON_PATH in the code and run again")
