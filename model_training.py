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
import dlib
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_FRAMES_PER_VIDEO = 25
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Face detector (dlib)
face_detector = dlib.get_frontal_face_detector()

# Load JSON dataset splits
def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# Set dataset paths
TRAIN_JSON = r"C:\Prof.Prakash\train.json"
VAL_JSON = r"C:\Prof.Prakash\val.json"
TEST_JSON = r"C:\Prof.Prakash\test.json"

REAL_DIR = r"C:\Prof.Prakash\original_sequences\youtube\c23\videos"
FAKE_DIR = r"C:\Prof.Prakash\manipulated_sequences\Deepfakes\c23\videos"

# Get full paths for videos
def get_video_paths(json_path, real_dir, fake_dir):
    entries = load_json(json_path)  # entries is a list of lists
    video_paths = []
    labels = []
    
    for entry in entries:
        if isinstance(entry, list) and len(entry) >= 2:
            # Original video ID
            original_id = entry[0]
            # Source video ID used for face swap
            source_id = entry[1]
            
            # Construct filenames
            real_filename = f"{original_id}.mp4"
            # Deepfake combines the target and source video IDs
            fake_filename = f"{original_id}_{source_id}.mp4"
            
            real_path = os.path.join(real_dir, real_filename)
            fake_path = os.path.join(fake_dir, fake_filename)
            
            if os.path.exists(real_path):
                video_paths.append(real_path)
                labels.append(0)  # 0 for real
            else:
                print(f"Warning: Real video {real_path} not found!")
                
            if os.path.exists(fake_path):
                video_paths.append(fake_path)
                labels.append(1)  # 1 for fake
            else:
                print(f"Warning: Fake video {fake_path} not found!")
        else:
            print(f"Warning: Unexpected entry format: {entry}")
            
    return video_paths, labels


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
        
        # Print model parameters summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

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

# Training function
def train(model, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    batch_times = []
    
    start_time = time.time()
    # Add tqdm progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
    
    
    scaler = GradScaler()
    
    for batch_idx, (frames, labels) in enumerate(progress_bar):
        batch_start = time.time()
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
        
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        # Update progress bar with current loss and accuracy
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%",
            'batch_time': f"{batch_times[-1]:.2f}s"
        })
    
    # Calculate metrics
    train_acc = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='weighted')
    train_recall = recall_score(all_labels, all_preds, average='weighted')
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    train_cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'accuracy': train_acc,
        'precision': train_precision,
        'recall': train_recall,
        'f1': train_f1,
        'confusion_matrix': train_cm,
        'avg_batch_time': sum(batch_times)/len(batch_times)
    }
    
    return metrics

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    start_time = time.time()
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
    val_precision = precision_score(all_labels, all_preds, average='weighted')
    val_recall = recall_score(all_labels, all_preds, average='weighted')
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'confusion_matrix': val_cm,
        'time': time.time() - start_time
    }
    
    return metrics

# Test function with detailed metrics
def test(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    start_time = time.time()
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(DEVICE)
            outputs = model(frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_cm = confusion_matrix(all_labels, all_preds)
    
    # Class-specific metrics
    class_precision = precision_score(all_labels, all_preds, average=None)
    class_recall = recall_score(all_labels, all_preds, average=None)
    class_f1 = f1_score(all_labels, all_preds, average=None)
    
    metrics = {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'confusion_matrix': test_cm,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'time': time.time() - start_time
    }
    
    return metrics, all_probs, all_labels, all_preds

# Plot metrics over epochs
def plot_training_progress(train_metrics, val_metrics, save_path="training_progress.png"):
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_metrics['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, val_metrics['loss'], 'ro-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_metrics['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_metrics['accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_metrics['f1'], 'bo-', label='Training F1')
    plt.plot(epochs, val_metrics['f1'], 'ro-', label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot Precision and Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_metrics['precision'], 'go-', label='Training Precision')
    plt.plot(epochs, val_metrics['precision'], 'g*-', label='Validation Precision')
    plt.plot(epochs, train_metrics['recall'], 'mo-', label='Training Recall')
    plt.plot(epochs, val_metrics['recall'], 'm*-', label='Validation Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True, path='best_model.pth'):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Main function
def main():
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION MODEL")
    print("="*50)
    
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Using device: {DEVICE}")
    print("Loading datasets...")
    
    # Create output directory
    output_dir = "deepfake_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video paths and labels
    train_videos, train_labels = get_video_paths(TRAIN_JSON, REAL_DIR, FAKE_DIR)
    val_videos, val_labels = get_video_paths(VAL_JSON, REAL_DIR, FAKE_DIR)
    test_videos, test_labels = get_video_paths(TEST_JSON, REAL_DIR, FAKE_DIR)
    
    # Dataset statistics
    train_real = train_labels.count(0)
    train_fake = train_labels.count(1)
    val_real = val_labels.count(0)
    val_fake = val_labels.count(1)
    test_real = test_labels.count(0)
    test_fake = test_labels.count(1)
    
    print("\nDATASET STATISTICS:")
    print(f"Training videos: {len(train_videos)} (Real: {train_real}, Fake: {train_fake})")
    print(f"Validation videos: {len(val_videos)} (Real: {val_real}, Fake: {val_fake})")
    print(f"Testing videos: {len(test_videos)} (Real: {test_real}, Fake: {test_fake})")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DeepfakeDataset(train_videos, train_labels)
    val_dataset = DeepfakeDataset(val_videos, val_labels)
    test_dataset = DeepfakeDataset(test_videos, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, pin_memory=True, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, pin_memory=True, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("\nCREATING MODEL:")
    print(f"Architecture: ResNeXt50 + BiLSTM")
    print(f"Frames per video: {NUM_FRAMES_PER_VIDEO}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    model = DeepfakeDetector().to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # For storing metrics
    train_history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    val_history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }

    early_stopping = EarlyStopping(
        patience=5,  # Stop if no improvement for 5 epochs
        verbose=True,
        path=os.path.join(output_dir, "early_stopping_model.pth")
    )
    
    scaler = GradScaler()

    # Training loop
    print("\nSTARTING TRAINING:")
    best_val_acc = 0
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)
        train_metrics = train(model, train_loader, criterion, optimizer, epoch, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion)
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        # Store metrics
        for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            train_history[metric].append(train_metrics[metric])
            val_history[metric].append(val_metrics[metric])
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print("\nEpoch Summary:")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}")
        
        # Print confusion matrix
        print("\nTraining Confusion Matrix:")
        print(train_metrics['confusion_matrix'])
        print("\nValidation Confusion Matrix:")
        print(val_metrics['confusion_matrix'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            print(f"\nNew best model! Saving at epoch {epoch+1}")
            best_model_path = os.path.join(output_dir, "best_deepfake_detector.pth")
            torch.save(model.state_dict(), best_model_path)
            
            # Also save checkpoint
            checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_acc': best_val_acc
            }, checkpoint_path)
        
        # Plot training progress
        plot_training_progress(train_history, val_history, 
                              os.path.join(output_dir, "training_progress.png"))

        # Early stopping
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    print(f"\nTraining complete. Loading best model from epoch {best_epoch+1} for testing...")
    best_model_path = os.path.join(output_dir, "best_deepfake_detector.pth")
    model.load_state_dict(torch.load(best_model_path))
    
    # Test
    print("\nEVALUATING ON TEST SET:")
    test_metrics, all_probs, all_labels, all_preds = test(model, test_loader)
    
    print("\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Time: {test_metrics['time']:.2f}s")
    
    print("\nClass-wise Metrics:")
    print(f"Real (Class 0) - Precision: {test_metrics['class_precision'][0]:.4f}, "
          f"Recall: {test_metrics['class_recall'][0]:.4f}, F1: {test_metrics['class_f1'][0]:.4f}")
    print(f"Fake (Class 1) - Precision: {test_metrics['class_precision'][1]:.4f}, "
          f"Recall: {test_metrics['class_recall'][1]:.4f}, F1: {test_metrics['class_f1'][1]:.4f}")
    
    print("\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    # Save final model and results
    final_model_path = os.path.join(output_dir, "final_deepfake_detector.pth")
    torch.save(model, final_model_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
        'train_history': train_history,
        'val_history': val_history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }, final_checkpoint_path)
    
    # Save results to text file
    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write("DEEPFAKE DETECTION MODEL SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"Base: ResNeXt50 + BiLSTM\n")
        f.write(f"Frames per video: {NUM_FRAMES_PER_VIDEO}\n")
        f.write(f"Image size: {IMAGE_SIZE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write(f"Training videos: {len(train_videos)} (Real: {train_real}, Fake: {train_fake})\n")
        f.write(f"Validation videos: {len(val_videos)} (Real: {val_real}, Fake: {val_fake})\n")
        f.write(f"Testing videos: {len(test_videos)} (Real: {test_real}, Fake: {test_fake})\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write(f"Best epoch: {best_epoch+1}/{NUM_EPOCHS}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n\n")
        
        f.write("TEST RESULTS:\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {test_metrics['f1']:.4f}\n\n")
        
        f.write("CLASS-WISE METRICS:\n")
        f.write(f"Real (Class 0) - Precision: {test_metrics['class_precision'][0]:.4f}, ")
        f.write(f"Recall: {test_metrics['class_recall'][0]:.4f}, F1: {test_metrics['class_f1'][0]:.4f}\n")
        f.write(f"Fake (Class 1) - Precision: {test_metrics['class_precision'][1]:.4f}, ")
        f.write(f"Recall: {test_metrics['class_recall'][1]:.4f}, F1: {test_metrics['class_f1'][1]:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"{test_metrics['confusion_matrix']}\n\n")
        
        f.write("EXECUTION TIME:\n")
        total_time = time.time() - start_time
        f.write(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n")
    
    # Print final timing information
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("\nTraining complete. All models and results saved to '{output_dir}' directory.")

if __name__ == "__main__":
    main()