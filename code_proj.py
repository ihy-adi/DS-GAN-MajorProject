import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import random
import time
from tqdm import tqdm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Data Loading and Processing

class SmokeDataset(Dataset):
    def __init__(self, smoky_dir, clear_dir, transform=None):
        """
        Dataset for smoke removal training
        
        Args:
            smoky_dir: Directory with smoky images
            clear_dir: Directory with corresponding clear images
            transform: Optional transforms to apply
        """
        self.smoky_dir = smoky_dir
        self.clear_dir = clear_dir
        self.transform = transform
        
        # Get filenames
        self.image_filenames = [f for f in os.listdir(smoky_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        smoky_path = os.path.join(self.smoky_dir, self.image_filenames[idx])
        clear_path = os.path.join(self.clear_dir, self.image_filenames[idx])
        
        smoky_image = Image.open(smoky_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")
        
        if self.transform:
            # Apply same transform to both images
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            smoky_image = self.transform(smoky_image)
            
            torch.manual_seed(seed)
            clear_image = self.transform(clear_image)
        
        return smoky_image, clear_image

class TestSmokeDataset(Dataset):
    """Dataset for testing smoke removal models with no ground truth clear images"""
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.test_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.image_filenames[idx]

# Data transforms
def get_transforms(img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

# Function to create data loaders
def create_dataloaders(smoky_dir, clear_dir, batch_size=8, img_size=256, test_split=0.2):
    transform = get_transforms(img_size)
    dataset = SmokeDataset(smoky_dir, clear_dir, transform)
    
    # Split into train and validation
    train_size = int((1 - test_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

#############################
# Model 1: U-Net for Smoke Removal
#############################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        p1 = self.pool(enc1)
        
        enc2 = self.enc2(p1)
        p2 = self.pool(enc2)
        
        enc3 = self.enc3(p2)
        p3 = self.pool(enc3)
        
        enc4 = self.enc4(p3)
        p4 = self.pool(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        
        # Decoder
        up1 = self.up1(bottleneck)
        concat1 = torch.cat([up1, enc4], dim=1)
        dec1 = self.dec1(concat1)
        
        up2 = self.up2(dec1)
        concat2 = torch.cat([up2, enc3], dim=1)
        dec2 = self.dec2(concat2)
        
        up3 = self.up3(dec2)
        concat3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.dec3(concat3)
        
        up4 = self.up4(dec3)
        concat4 = torch.cat([up4, enc1], dim=1)
        dec4 = self.dec4(concat4)
        
        # Output
        output = self.final_conv(dec4)
        output = self.sigmoid(output)
        
        return output

#############################
# Model 2: DehazeNet for Smoke Removal
#############################

class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Multi-scale mapping
        self.conv_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_7x7 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        
        # Feature fusion
        self.fusion = nn.Conv2d(64*3, 64, kernel_size=1)
        
        # Transmission map estimation
        self.trans_est = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Reconstruction layers
        self.recon = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feature extraction
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        
        # Multi-scale mapping
        y1 = F.relu(self.conv_3x3(x3))
        y2 = F.relu(self.conv_5x5(x3))
        y3 = F.relu(self.conv_7x7(x3))
        
        # Feature fusion
        z = torch.cat([y1, y2, y3], dim=1)
        z = F.relu(self.fusion(z))
        
        # Transmission map estimation
        trans = self.trans_est(z)
        
        # Combine transmission map with input for reconstruction
        combined = torch.cat([x, trans.expand(-1, 1, -1, -1)], dim=1)
        
        # Reconstruction
        output = self.recon(combined)
        
        return output

#############################
# Model 3: GAN for Smoke Removal (based on the paper by N Bharath Raj)
#############################

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder (upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Decoder with skip connections
        d1 = self.dec1(e5)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        output = self.dec5(d4)
        
        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 128 x 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 16 x 16
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            # 1 x 15 x 15
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

#############################
# Model 4: CycleGAN for Unpaired Smoke Removal
#############################

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

class CycleGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(CycleGANDiscriminator, self).__init__()
        
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # FCN classification layer
        model += [nn.Conv2d(512, 1, kernel_size=4, padding=1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

#############################
# Human Detection Models
#############################

class YOLOv5Wrapper:
    def __init__(self, model_size='s'):
        """
        A wrapper for YOLOv5 model
        
        Args:
            model_size: YOLOv5 model size ('n', 's', 'm', 'l', 'x')
        """
        self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}')
        self.model.classes = [0]  # Only detect persons (class 0 in COCO)
        
    def detect(self, img, conf_threshold=0.25):
        """
        Detect people in an image
        
        Args:
            img: Image as numpy array (BGR)
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(img)
        detections = results.pandas().xyxy[0]
        
        # Filter for person class (0) and confidence threshold
        filtered = detections[(detections['class'] == 0) & (detections['confidence'] >= conf_threshold)]
        
        # Convert to list of [x1, y1, x2, y2, confidence, class_id]
        detections_list = []
        for _, row in filtered.iterrows():
            detections_list.append([
                row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                row['confidence'], row['class']
            ])
            
        return detections_list

class FasterRCNNWrapper:
    def __init__(self, pretrained=True):
        """
        A wrapper for Faster R-CNN model
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        self.model.eval()
        self.model.to(device)
        
        # COCO class index for person is 1
        self.person_class_idx = 1
        
    def detect(self, img, conf_threshold=0.5):
        """
        Detect people in an image
        
        Args:
            img: Image as numpy array (BGR)
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
            
        # Extract person detections
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter for person class and confidence threshold
        person_mask = (labels == self.person_class_idx) & (scores >= conf_threshold)
        person_boxes = boxes[person_mask]
        person_scores = scores[person_mask]
        
        # Convert to list of [x1, y1, x2, y2, confidence, class_id]
        detections = []
        for box, score in zip(person_boxes, person_scores):
            detections.append([
                box[0], box[1], box[2], box[3],
                score, 0  # Using 0 as person class_id to standardize with YOLO
            ])
            
        return detections

#############################
# Training Functions
#############################

def train_unet(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """Train U-Net for smoke removal"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for smoky_imgs, clear_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            smoky_imgs = smoky_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(smoky_imgs)
            loss = criterion(outputs, clear_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * smoky_imgs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for smoky_imgs, clear_imgs in val_loader:
                smoky_imgs = smoky_imgs.to(device)
                clear_imgs = clear_imgs.to(device)
                
                outputs = model(smoky_imgs)
                loss = criterion(outputs, clear_imgs)
                
                val_loss += loss.item() * smoky_imgs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"Model saved with Val Loss: {val_loss:.4f}")

def train_dehazenet(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """Train DehazeNet for smoke removal"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for smoky_imgs, clear_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            smoky_imgs = smoky_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(smoky_imgs)
            loss = criterion(outputs, clear_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * smoky_imgs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for smoky_imgs, clear_imgs in val_loader:
                smoky_imgs = smoky_imgs.to(device)
                clear_imgs = clear_imgs.to(device)
                
                outputs = model(smoky_imgs)
                loss = criterion(outputs, clear_imgs)
                
                val_loss += loss.item() * smoky_imgs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_dehazenet_model.pth')
            print(f"Model saved with Val Loss: {val_loss:.4f}")

def train_gan(generator, discriminator, train_loader, val_loader, num_epochs=50, lr=0.0002):
    """Train GAN for smoke removal"""
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    generator.to(device)
    discriminator.to(device)
    
    # Lambda for pixel-wise loss
    lambda_pixel = 100
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        for i, (smoky_imgs, clear_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            smoky_imgs = smoky_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            # Ground truths
            valid = torch.ones((smoky_imgs.size(0), 1, 15, 15), requires_grad=False).to(device)
            fake = torch.zeros((smoky_imgs.size(0), 1, 15, 15), requires_grad=False).to(device)
            
            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            
            # Generate a batch of images
            gen_imgs = generator(smoky_imgs)
            
            # Adversarial loss
            pred_fake = discriminator(gen_imgs)
            loss_gan = criterion_gan(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixel(gen_imgs, clear_imgs)
            
            # Total loss
            loss_G = loss_gan + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
            
            # ------------------
            #  Train Discriminator
            # ------------------
            optimizer_D.zero_grad()
            
            # Real images
            pred_real = discriminator(clear_imgs)
            loss_real = criterion_gan(pred_real, valid)
            
            # Fake images
            pred_fake = discriminator(gen_imgs.detach())
            loss_fake = criterion_gan(pred_fake, fake)
            
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            
            if i % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"[Batch {i}/{len(train_loader)}] "
                    f"[D loss: {loss_D.item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}]"
                )
        
        # Validation
        generator.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for smoky_imgs, clear_imgs in val_loader:
                smoky_imgs = smoky_imgs.to(device)
                clear_imgs = clear_imgs.to(device)
                
                gen_imgs = generator(smoky_imgs)
                loss_pixel = criterion_pixel(gen_imgs, clear_imgs)
                val_loss += loss_pixel.item() * smoky_imgs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save model
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

def train_cyclegan(G_A2B, G_B2A, D_A, D_B, train_loader, num_epochs=100, lr=0.0002):
    """
    Train CycleGAN for smoke removal
    G_A2B: Generator that transforms smoky images (A) to clear images (B)
    G_B2A: Generator that transforms clear images (B) to smoky images (A)
    D_A: Discriminator for smoky images (A)
    D_B: Discriminator for clear images (B)
    """
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    optimizer_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    
    G_A2B.to(device)
    G_B2A.to(device)
    D_A.to(device)
    D_B.to(device)
    
    # Weights for different loss components
    lambda_cycle = 10.0
    lambda_identity = 5.0
    
    for epoch in range(num_epochs):
        G_A2B.train()
        G_B2A.train()
        D_A.train()
        D_B.train()
        
        for i, (smoky_imgs, clear_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            smoky_imgs = smoky_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            # Ground truths
            valid = torch.ones((smoky_imgs.size(0), 1, 30, 30), requires_grad=False).to(device)
            fake = torch.zeros((smoky_imgs.size(0), 1, 30, 30), requires_grad=False).to(device)
            
            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_A = criterion_identity(G_B2A(smoky_imgs), smoky_imgs)
            loss_id_B = criterion_identity(G_A2B(clear_imgs), clear_imgs)
            loss_identity = (loss_id_A + loss_id_B) * lambda_identity
            
            # GAN loss
            fake_B = G_A2B(smoky_imgs)
            loss_GAN_A2B = criterion_gan(D_B(fake_B), valid)
            
            fake_A = G_B2A(clear_imgs)
            loss_GAN_B2A = criterion_gan(D_A(fake_A), valid)
            
            loss_GAN = loss_GAN_A2B + loss_GAN_B2A
            
            # Cycle loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, smoky_imgs)
            
            recovered_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, clear_imgs)
            
            loss_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle
            
            # Total loss
            loss_G = loss_identity + loss_GAN + loss_cycle
            loss_G.backward()
            optimizer_G.step()
            
            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            
            # Real loss
            loss_real = criterion_gan(D_A(smoky_imgs), valid)
            # Fake loss (using detached fake_A)
            loss_fake = criterion_gan(D_A(fake_A.detach()), fake)
            
            loss_D_A = (loss_real + loss_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            
            # Real loss
            loss_real = criterion_gan(D_B(clear_imgs), valid)
            # Fake loss (using detached fake_B)
            loss_fake = criterion_gan(D_B(fake_B.detach()), fake)
            
            loss_D_B = (loss_real + loss_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            
            if i % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"[Batch {i}/{len(train_loader)}] "
                    f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}]"
                )
        
        # Save models
        if (epoch + 1) % 10 == 0:
            torch.save(G_A2B.state_dict(), f'G_A2B_epoch_{epoch+1}.pth')
            torch.save(G_B2A.state_dict(), f'G_B2A_epoch_{epoch+1}.pth')
            torch.save(D_A.state_dict(), f'D_A_epoch_{epoch+1}.pth')
            torch.save(D_B.state_dict(), f'D_B_epoch_{epoch+1}.pth')

#############################
# Smoke Removal and Human Detection Pipeline
#############################

class SmokeRemovalHumanDetection:
    def __init__(self, smoke_removal_model_type='unet', human_detection_model_type='yolov5'):
        """
        Initialize the pipeline with the specified models
        
        Args:
            smoke_removal_model_type: Type of smoke removal model ('unet', 'dehazenet', 'gan', 'cyclegan')
            human_detection_model_type: Type of human detection model ('yolov5', 'fasterrcnn')
        """
        self.smoke_removal_model_type = smoke_removal_model_type
        self.human_detection_model_type = human_detection_model_type
        
        # Initialize smoke removal model
        if smoke_removal_model_type == 'unet':
            self.smoke_removal_model = UNet()
        elif smoke_removal_model_type == 'dehazenet':
            self.smoke_removal_model = DehazeNet()
        elif smoke_removal_model_type == 'gan':
            self.smoke_removal_model = Generator()
        elif smoke_removal_model_type == 'cyclegan':
            self.smoke_removal_model = CycleGANGenerator()
        else:
            raise ValueError("Unsupported smoke removal model type")
        
        # Initialize human detection model
        if human_detection_model_type == 'yolov5':
            self.human_detection_model = YOLOv5Wrapper()
        elif human_detection_model_type == 'fasterrcnn':
            self.human_detection_model = FasterRCNNWrapper()
        else:
            raise ValueError("Unsupported human detection model type")
        
        # Load model weights if available
        try:
            if smoke_removal_model_type == 'unet':
                self.smoke_removal_model.load_state_dict(torch.load('best_unet_model.pth'))
            elif smoke_removal_model_type == 'dehazenet':
                self.smoke_removal_model.load_state_dict(torch.load('best_dehazenet_model.pth'))
            elif smoke_removal_model_type == 'gan':
                self.smoke_removal_model.load_state_dict(torch.load('generator_epoch_50.pth'))
            elif smoke_removal_model_type == 'cyclegan':
                self.smoke_removal_model.load_state_dict(torch.load('G_A2B_epoch_100.pth'))
                
            self.smoke_removal_model.to(device)
            self.smoke_removal_model.eval()
            print(f"Loaded smoke removal model: {smoke_removal_model_type}")
        except:
            print(f"Could not load smoke removal model weights. Using untrained model.")
        
    def process_image(self, image, detection_threshold=0.5):
        """
        Process image with smoke removal and human detection
        
        Args:
            image: Input image (numpy array in BGR format)
            detection_threshold: Confidence threshold for human detection
            
        Returns:
            Processed image with smoke removed and human detections
        """
        # Record time
        start_time = time.time()
        
        # Prepare image for smoke removal model
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Apply smoke removal
        with torch.no_grad():
            output = self.smoke_removal_model(img_tensor)
        
        # Convert output back to image
        output = output.squeeze().cpu().detach()
        output = output * 0.5 + 0.5  # Denormalize
        output = output.clamp(0, 1)
        output = output.permute(1, 2, 0).numpy() * 255
        output = cv2.resize(output, (w, h))
        output = output.astype(np.uint8)
        
        # Convert back to BGR for OpenCV
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Apply human detection
        detections = self.human_detection_model.detect(output_bgr, conf_threshold=detection_threshold)
        
        # Draw bounding boxes on the image
        for box in detections:
            x1, y1, x2, y2, conf, class_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(output_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_bgr, f"Person: {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate processing time
        process_time = time.time() - start_time
        cv2.putText(output_bgr, f"Time: {process_time:.3f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return output_bgr
    
    def process_video(self, input_video_path, output_video_path, detection_threshold=0.5):
        """
        Process video with smoke removal and human detection
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            detection_threshold: Confidence threshold for human detection
        """
        cap = cv2.VideoCapture(input_video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Process each frame
        pbar = tqdm(total=frame_count, desc="Processing video")
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_image(frame, detection_threshold)
            
            # Write processed frame
            out.write(processed_frame)
            
            frame_idx += 1
            pbar.update(1)
            
            # Display progress
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        # Release resources
        cap.release()
        out.release()
        pbar.close()
        print(f"Video processing complete. Output saved to: {output_video_path}")

#############################
# Evaluation Functions
#############################

def evaluate_smoke_removal(model, test_loader):
    """
    Evaluate smoke removal model using PSNR and SSIM metrics
    
    Args:
        model: Smoke removal model
        test_loader: DataLoader for test dataset
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for smoky_imgs, clear_imgs in tqdm(test_loader, desc="Evaluating smoke removal"):
            smoky_imgs = smoky_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            # Generate output
            output = model(smoky_imgs)
            
            # Calculate PSNR and SSIM
            for i in range(smoky_imgs.size(0)):
                # Denormalize and convert to numpy
                output_img = output[i].cpu().detach().numpy().transpose(1, 2, 0)
                output_img = (output_img * 0.5 + 0.5) * 255
                output_img = np.clip(output_img, 0, 255).astype(np.uint8)
                
                clear_img = clear_imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
                clear_img = (clear_img * 0.5 + 0.5) * 255
                clear_img = np.clip(clear_img, 0, 255).astype(np.uint8)
                
                # Calculate PSNR
                mse = np.mean((output_img - clear_img) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                psnr_values.append(psnr)
                
                # Calculate SSIM
                ssim = cv2.compareSSIM(output_img, clear_img, multichannel=True)
                ssim_values.append(ssim)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

def evaluate_human_detection(model, test_loader, ground_truth_annotations):
    """
    Evaluate human detection model using precision, recall, and F1 score
    
    Args:
        model: Human detection model
        test_loader: DataLoader for test dataset
        ground_truth_annotations: Dictionary of ground truth annotations
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for images, image_names in tqdm(test_loader, desc="Evaluating human detection"):
        for i, image in enumerate(images):
            image_name = image_names[i]
            
            # Convert tensor to numpy array
            image = image.cpu().detach().numpy().transpose(1, 2, 0)
            image = (image * 0.5 + 0.5) * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get detections
            detections = model.detect(image)
            
            # Get ground truth boxes
            gt_boxes = ground_truth_annotations.get(image_name, [])
            
            # Calculate IoU for each detection with ground truth
            matched_gt = set()
            
            for detection in detections:
                det_box = detection[:4]  # [x1, y1, x2, y2]
                
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                        
                    # Calculate IoU
                    iou = calculate_iou(det_box, gt_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                # If IoU > 0.5, consider it a match
                if best_iou > 0.5:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            # Unmatched ground truth boxes are false negatives
            false_negatives += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return precision, recall, f1

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

#############################
# Main Function
#############################

def main():
    """Main function to run the smoke removal and human detection pipeline"""
    parser = argparse.ArgumentParser(description="Smoke Removal and Human Detection")
    parser.add_argument("--smoke_removal_model", type=str, default="unet", 
                        choices=["unet", "dehazenet", "gan", "cyclegan"],
                        help="Type of smoke removal model")
    parser.add_argument("--human_detection_model", type=str, default="yolov5",
                        choices=["yolov5", "fasterrcnn"],
                        help="Type of human detection model")
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the models")
    parser.add_argument("--process_image", type=str, help="Process a single image")
    parser.add_argument("--process_video", type=str, help="Process a video")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--smoky_dir", type=str, help="Directory with smoky images for training")
    parser.add_argument("--clear_dir", type=str, help="Directory with clear images for training")
    parser.add_argument("--test_dir", type=str, help="Directory with test images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Train models if requested
    if args.train:
        if args.smoke_removal_model == "unet":
            print("Training U-Net model...")
            model = UNet()
            train_loader, val_loader = create_dataloaders(args.smoky_dir, args.clear_dir, args.batch_size)
            train_unet(model, train_loader, val_loader, args.epochs, args.lr)
        
        elif args.smoke_removal_model == "dehazenet":
            print("Training DehazeNet model...")
            model = DehazeNet()
            train_loader, val_loader = create_dataloaders(args.smoky_dir, args.clear_dir, args.batch_size)
            train_dehazenet(model, train_loader, val_loader, args.epochs, args.lr)
        
        elif args.smoke_removal_model == "gan":
            print("Training GAN model...")
            generator = Generator()
            discriminator = Discriminator()
            train_loader, val_loader = create_dataloaders(args.smoky_dir, args.clear_dir, args.batch_size)
            train_gan(generator, discriminator, train_loader, val_loader, args.epochs, args.lr)
        
        elif args.smoke_removal_model == "cyclegan":
            print("Training CycleGAN model...")
            G_A2B = CycleGANGenerator()  # Smoky to clear
            G_B2A = CycleGANGenerator()  # Clear to smoky
            D_A = CycleGANDiscriminator()  # Discriminator for smoky images
            D_B = CycleGANDiscriminator()  # Discriminator for clear images
            train_loader, val_loader = create_dataloaders(args.smoky_dir, args.clear_dir, args.batch_size)
            train_cyclegan(G_A2B, G_B2A, D_A, D_B, train_loader, args.epochs, args.lr)
    
    # Create the pipeline
    pipeline = SmokeRemovalHumanDetection(
        smoke_removal_model_type=args.smoke_removal_model,
        human_detection_model_type=args.human_detection_model
    )
    
    # Process a single image if requested
    if args.process_image:
        print(f"Processing image: {args.process_image}")
        image = cv2.imread(args.process_image)
        if image is None:
            print(f"Error: Could not read image {args.process_image}")
            return
        
        processed_image = pipeline.process_image(image)
        output_path = os.path.join(args.output, "processed_" + os.path.basename(args.process_image))
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved to: {output_path}")
    
    # Process a video if requested
    if args.process_video:
        print(f"Processing video: {args.process_video}")
        output_path = os.path.join(args.output, "processed_" + os.path.basename(args.process_video))
        pipeline.process_video(args.process_video, output_path)

if __name__ == "__main__":
    main()
