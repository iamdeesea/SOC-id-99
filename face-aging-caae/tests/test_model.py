"""
Unit tests for CAAE model.
"""

import unittest
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.caae import CAAE, Encoder, Decoder
from utils.preprocessing import ImagePreprocessor


class TestCAAE(unittest.TestCase):
    """Test cases for CAAE model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CAAE()
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 128, 128)
        self.age_condition = torch.randn(self.batch_size, 6)
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = Encoder()
        z, c = encoder(self.input_tensor)
        
        self.assertEqual(z.shape, (self.batch_size, 50))
        self.assertEqual(c.shape, (self.batch_size, 6))
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = Decoder()
        z = torch.randn(self.batch_size, 50)
        c = torch.randn(self.batch_size, 6)
        
        output = decoder(z, c)
        self.assertEqual(output.shape, (self.batch_size, 3, 128, 128))
    
    def test_caae_forward(self):
        """Test complete CAAE forward pass."""
        output, z, c = self.model(self.input_tensor)
        
        self.assertEqual(output.shape, self.input_tensor.shape)
        self.assertEqual(z.shape, (self.batch_size, 50))
        self.assertEqual(c.shape, (self.batch_size, 6))
    
    def test_caae_forward_with_target_age(self):
        """Test CAAE forward pass with target age conditioning."""
        output, z, c = self.model(self.input_tensor, self.age_condition)
        
        self.assertEqual(output.shape, self.input_tensor.shape)
        self.assertEqual(z.shape, (self.batch_size, 50))
        self.assertEqual(c.shape, (self.batch_size, 6))


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for image preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        
        # Create test image
        self.test_image = Image.new('RGB', (256, 256), color='red')
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        tensor_image, pil_image = self.preprocessor.preprocess_image(self.test_image)
        
        self.assertEqual(tensor_image.shape, (3, 128, 128))
        self.assertIsInstance(pil_image, Image.Image)
    
    def test_face_detection(self):
        """Test face detection (may not find faces in synthetic images)."""
        face = self.preprocessor.detect_face(self.test_image)
        self.assertIsInstance(face, np.ndarray)
        self.assertEqual(len(face.shape), 3)  # Should be RGB image


if __name__ == '__main__':
    unittest.main()
