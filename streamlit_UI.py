import streamlit as st
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import tempfile
import dlib
import time
from torch.cuda.amp import autocast
import torch.nn as nn
from torchvision import models

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="centered"
)

# Constants
IMAGE_SIZE = (224, 224)
NUM_FRAMES_PER_VIDEO = 25
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to the saved model
MODEL_PATH = "C:\\Users\\yuvra\\OneDrive\\Desktop\\ML Labs\\Projects\\Deepfake Detection\\final_deepfake_detector.pth"

# Face detector
@st.cache_resource
def load_face_detector():
    return dlib.get_frontal_face_detector()

face_detector = load_face_detector()

# Define the model architecture (same as in your training script)
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

# Load the trained model
@st.cache_resource
def load_model(model_path):
    try:
        model = DeepfakeDetector(num_classes=2).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None



# Function to extract frames from video
def extract_frames(video_path, num_frames=NUM_FRAMES_PER_VIDEO):
    frames = []
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file.")
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        st.error(f"Error: No frames in video.")
        cap.release()
        return None
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            face = detect_and_crop_face(frame)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face_tensor = transform(face)
            frames.append(face_tensor)
        else:
            frames.append(torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

    cap.release()
    
    # Ensure we have exactly num_frames
    if len(frames) < num_frames:
        for _ in range(num_frames - len(frames)):
            frames.append(torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    
    return torch.stack(frames[:num_frames])

# Function to detect and crop face
def detect_and_crop_face(frame):
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
        st.error(f"Error in face detection: {e}")
        return cv2.resize(frame, IMAGE_SIZE)

# Function to predict if a video is deepfake or not
def predict_deepfake(video_path, model):
    frames = extract_frames(video_path)
    if frames is None:
        return None, None

    frames = frames.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
    
    with torch.no_grad():
        with autocast():
            outputs = model(frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
    
    return predicted.item(), probabilities[0].cpu().numpy()

# Function to display the prediction result
def display_prediction(prediction, probabilities):
    if prediction == 1:  # Fake
        st.error("⚠️ DEEPFAKE DETECTED ⚠️")
        st.write(f"Confidence: {probabilities[1]*100:.2f}%")
    else:  # Real
        st.success("✅ AUTHENTIC VIDEO")
        st.write(f"Confidence: {probabilities[0]*100:.2f}%")
    
    # Display probability bar chart
    st.write("Prediction Probabilities:")
    chart_data = {
        "Category": ["Real", "Deepfake"],
        "Probability": [float(probabilities[0]), float(probabilities[1])]
    }
    
    st.bar_chart(
        chart_data,
        x="Category",
        y="Probability", 
        color="#ffaa00"
    )

# Streamlit UI
def main():
    
    
    st.title("🔍 Deepfake Detection System")
    st.write("Upload a video to detect if it's a deepfake")
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(f"Failed to load model from {MODEL_PATH}. Please check if the model file exists.")
        st.info("To use this app, make sure you have a trained model saved at the specified path.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file (.mp4)", type=["mp4"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # Display the uploaded video
        st.video(video_path)
        
        # Process button
        if st.button("Analyze Video"):
            with st.spinner("Analyzing video... This may take a minute."):
                start_time = time.time()
                prediction, probabilities = predict_deepfake(video_path, model)
                processing_time = time.time() - start_time
                
                if prediction is not None:
                    st.write(f"⏱️ Processing time: {processing_time:.2f} seconds")
                    display_prediction(prediction, probabilities)
                    
                    # Display technical details
                    with st.expander("Technical Details"):
                        st.write(f"• Number of frames analyzed: {NUM_FRAMES_PER_VIDEO}")
                        st.write(f"• Frame resolution: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
                        st.write(f"• Model architecture: ResNeXt50 + BiLSTM")
                        st.write(f"• Processing device: {DEVICE}")
                        st.write(f"• Raw probability scores: Real: {probabilities[0]:.4f}, Fake: {probabilities[1]:.4f}")
                else:
                    st.error("Failed to process the video. Please try with a different video.")
        
        # Clean up the temporary file
        os.unlink(video_path)
    
    st.divider()
    st.markdown("""
    ### How it works
    
    This application uses a deep learning model combining ResNeXt50 CNN with bidirectional LSTM to analyze facial features 
    across multiple frames of the video. The model has been trained on the DeepFake Detection Challenge dataset to 
    distinguish between authentic videos and manipulated deepfake videos.
    
    ### Tips for best results
    - Videos should have at least one visible face
    - Higher quality videos typically yield more accurate results
    - Front-facing angles work better than profiles
    """)

if __name__ == "__main__":
    main()
