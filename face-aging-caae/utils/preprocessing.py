"""
Image preprocessing utilities for face aging.
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    """Handles image preprocessing for face aging tasks."""
    
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def detect_face(self, image):
        """Detect and crop face from image."""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = np.array(image)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            
            # Add padding
            padding = int(0.2 * max(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            face = img[y:y+h, x:x+w]
            return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            # If no face detected, return original image
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def preprocess_image(self, image_path_or_array):
        """Complete preprocessing pipeline."""
        # Detect and crop face
        face = self.detect_face(image_path_or_array)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face)
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        
        return tensor_image, pil_image
