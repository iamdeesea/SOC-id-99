"""
Flask API for Face Aging service.
"""

from flask import Flask, request, jsonify, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.caae import CAAE
from utils.preprocessing import ImagePreprocessor

app = Flask(__name__)

# Global variables
model = None
preprocessor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_model():
    """Initialize the face aging model."""
    global model, preprocessor
    
    model = CAAE()
    model_path = os.environ.get('MODEL_PATH', 'models/face_aging_model.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("No pre-trained model found. Using randomly initialized weights.")
    
    model.to(device)
    model.eval()
    
    preprocessor = ImagePreprocessor()
    print("Model initialized successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/age_face', methods=['POST'])
def age_face_api():
    """Main face aging endpoint."""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        target_age = int(request.form.get('target_age', 2))
        
        if target_age < 0 or target_age > 5:
            return jsonify({'error': 'target_age must be between 0 and 5'}), 400
        
        # Process image
        image = Image.open(image_file.stream)
        aged_image = age_face_function(image, target_age)
        
        # Convert result to base64
        buffered = io.BytesIO()
        aged_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'aged_image': img_str,
            'target_age_group': target_age,
            'age_groups': ['0-10', '11-20', '21-30', '31-40', '41-50', '51+'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_age', methods=['POST'])
def batch_age_api():
    """Batch processing endpoint."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        images = request.files.getlist('images')
        target_age = int(request.form.get('target_age', 2))
        
        results = []
        for i, image_file in enumerate(images):
            try:
                image = Image.open(image_file.stream)
                aged_image = age_face_function(image, target_age)
                
                # Convert to base64
                buffered = io.BytesIO()
                aged_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                results.append({
                    'index': i,
                    'filename': image_file.filename,
                    'aged_image': img_str,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'filename': image_file.filename,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return jsonify({
            'results': results,
            'target_age_group': target_age,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def age_face_function(image, target_age):
    """Core face aging function."""
    global model, preprocessor
    
    # Preprocess image
    tensor_image, _ = preprocessor.preprocess_image(image)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    
    # Create age condition
    age_condition = torch.zeros(1, 6).to(device)
    age_condition[0, target_age] = 1.0
    
    with torch.no_grad():
        # Get latent representation
        z, _ = model.encoder(tensor_image)
        
        # Generate aged image
        aged_tensor = model.decoder(z, age_condition)
        
        # Convert back to PIL image
        aged_tensor = aged_tensor.squeeze(0).cpu()
        aged_tensor = (aged_tensor + 1) / 2  # Denormalize
        aged_tensor = torch.clamp(aged_tensor, 0, 1)
        aged_image = transforms.ToPILImage()(aged_tensor)
    
    return aged_image


if __name__ == '__main__':
    initialize_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
