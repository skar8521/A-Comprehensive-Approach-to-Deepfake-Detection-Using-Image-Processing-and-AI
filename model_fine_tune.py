import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import dlib
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_FRAMES_PER_VIDEO = 25
LEARNING_RATE = 0.0001  # Slightly lower learning rate since we're fine-tuning
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Face detector (dlib)
face_detector = dlib.get_frontal_face_detector()

# Define paths
CELEB_REAL_DIR = "/kaggle/input/celeb-df-v2/Celeb-real"
CELEB_FAKE_DIR = "/kaggle/input/celeb-df-v2/Celeb-synthesis"
YOUTUBE_REAL_DIR = "/kaggle/input/celeb-df-v2/YouTube-real"
TEST_LIST_PATH = "/kaggle/input/celeb-df-v2/List_of_testing_videos.txt"
MODEL_PATH = "/kaggle/input/final_deepfake_detection/pytorch/default/1/final_deepfake_detector.pth"
OUTPUT_DIR = "/kaggle/working/celeb_df_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model: ResNeXt-50 with LSTM
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        # Load pretrained ResNeXt but remove the final fully connected layer
        self.resnext = models.resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.resnext.children())[:-1])
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(2048, 512, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        
        # Reshape for CNN
        x = x.view(batch_size * timesteps, C, H, W)
        
        # Extract features using ResNeXt
        x = self.feature_extractor(x)
        x = x.view(batch_size, timesteps, -1)
        
        # Process with LSTM
        x, _ = self.lstm(x)
        
        # Take the final timestep's output
        x = self.fc(x[:, -1, :])
        
        return x

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=NUM_FRAMES_PER_VIDEO):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return torch.zeros(self.num_frames, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
            
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Error: No frames in video: {video_path}")
            cap.release()
            return torch.zeros(self.num_frames, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
            
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                face = self._detect_and_crop_face(frame)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
                face_tensor = self.transform(face)
                frames.append(face_tensor)
            else:
                frames.append(torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        cap.release()
        
        # Ensure we have exactly num_frames
        if len(frames) < self.num_frames:
            for _ in range(self.num_frames - len(frames)):
                frames.append(torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        
        return torch.stack(frames[:self.num_frames])

    def _detect_and_crop_face(self, frame):
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            
            if faces:
                # Get the largest face
                largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                x, y, w, h = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()
                
                # Add padding
                padding = int(max(w, h) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                cropped_face = frame[y:y+h, x:x+w]
                return cv2.resize(cropped_face, IMAGE_SIZE)
            else:
                # If no face is detected, use the center of the frame
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                start_x = (w - min_dim) // 2
                start_y = (h - min_dim) // 2
                cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
                return cv2.resize(cropped, IMAGE_SIZE)
        except Exception as e:
            print(f"Error in face detection: {e}")
            return cv2.resize(frame, IMAGE_SIZE)

# Load test list to exclude from training
def load_test_videos():
    test_videos = []
    with open(TEST_LIST_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 2)
            if len(parts) >= 3:
                test_videos.append(parts[2])
    return test_videos

# Load all videos from CelebDF dataset
def load_celeb_df_videos():
    test_videos = load_test_videos()
    all_videos = []
    all_labels = []
    
    # Add Celeb-real videos (real)
    print("Loading Celeb-real videos...")
    for video_name in os.listdir(CELEB_REAL_DIR):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(CELEB_REAL_DIR, video_name)
            rel_path = f"Celeb-real/{video_name}"
            
            # Skip if this is a test video
            if rel_path in test_videos:
                continue
                
            all_videos.append(video_path)
            all_labels.append(0)  # 0 for real
    
    # Add Celeb-synthesis videos (fake)
    print("Loading Celeb-synthesis videos...")
    for video_name in os.listdir(CELEB_FAKE_DIR):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(CELEB_FAKE_DIR, video_name)
            rel_path = f"Celeb-synthesis/{video_name}"
            
            # Skip if this is a test video
            if rel_path in test_videos:
                continue
                
            all_videos.append(video_path)
            all_labels.append(1)  # 1 for fake
    
    # Add YouTube-real videos (real)
    print("Loading YouTube-real videos...")
    for video_name in os.listdir(YOUTUBE_REAL_DIR):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(YOUTUBE_REAL_DIR, video_name)
            rel_path = f"YouTube-real/{video_name}"
            
            # Skip if this is a test video
            if rel_path in test_videos:
                continue
                
            all_videos.append(video_path)
            all_labels.append(0)  # 0 for real
    
    return all_videos, all_labels

# Load test videos from the test list
def load_test_dataset():
    test_videos = []
    test_labels = []
    
    with open(TEST_LIST_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 2)
            if len(parts) >= 3:
                label = int(parts[0])
                video_path = parts[2]
                
                # Determine the absolute path
                if video_path.startswith('Celeb-real/'):
                    full_path = os.path.join(CELEB_REAL_DIR, video_path.split('/', 1)[1])
                elif video_path.startswith('Celeb-synthesis/'):
                    full_path = os.path.join(CELEB_FAKE_DIR, video_path.split('/', 1)[1])
                elif video_path.startswith('YouTube-real/'):
                    full_path = os.path.join(YOUTUBE_REAL_DIR, video_path.split('/', 1)[1])
                else:
                    continue
                    
                test_videos.append(full_path)
                test_labels.append(0 if label == 1 else 1)  # Our model uses 0 for real and 1 for fake
    
    return test_videos, test_labels

# Training function
def train(model, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
    
    for batch_idx, (frames, labels) in enumerate(progress_bar):
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Store for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    # Calculate metrics
    train_acc = accuracy_score(all_labels, all_preds)
    
    return train_acc, total_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_acc, total_loss / len(val_loader)

# Test function
def test(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Testing"):
            frames = frames.to(DEVICE)
            outputs = model(frames)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    
    return test_acc

def main():
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION MODEL TRAINING ON CELEB-DF")
    print("="*50)
    
    # Load model
    print("Loading pretrained model...")
    model = DeepfakeDetector().to(DEVICE)
    loaded_model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(loaded_model.state_dict())
    
    # Load videos
    print("Loading video datasets...")
    all_videos, all_labels = load_celeb_df_videos()
    
    # Split into train and validation
    print("Creating train/validation split...")
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        all_videos, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # Load test dataset
    print("Loading test dataset...")
    test_videos, test_labels = load_test_dataset()
    
    # Print dataset statistics
    print("\nDATASET STATISTICS:")
    print(f"Training videos: {len(train_videos)} (Real: {train_labels.count(0)}, Fake: {train_labels.count(1)})")
    print(f"Validation videos: {len(val_videos)} (Real: {val_labels.count(0)}, Fake: {val_labels.count(1)})")
    print(f"Testing videos: {len(test_videos)} (Real: {test_labels.count(0)}, Fake: {test_labels.count(1)})")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DeepfakeDataset(train_videos, train_labels)
    val_dataset = DeepfakeDataset(val_videos, val_labels)
    test_dataset = DeepfakeDataset(test_videos, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    scaler = GradScaler()
    
    # Training loop
    print("\nSTARTING TRAINING:")
    best_val_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, epoch, scaler)
        
        # Validate
        val_acc, val_loss = validate(model, val_loader, criterion)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print results
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    
    # Test
    test_acc = test(model, test_loader)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    print(f"Final model saved to {os.path.join(OUTPUT_DIR, 'final_model.pth')}")

if __name__ == "__main__":
    main()