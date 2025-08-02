"""
Face Aging CAAE Models Module

This module contains the implementation of the Conditional Adversarial Autoencoder (CAAE)
for face aging tasks.
"""

from .caae import CAAE, Encoder, Decoder
from .modified_caae import ModifiedCAAE

__all__ = ['CAAE', 'Encoder', 'Decoder', 'ModifiedCAAE']
