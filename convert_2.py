import tensorflow as tf
from safetensors.torch import load_file
import torch
from sentence_transformers import SentenceTransformer

# Load the .safetensors file
tensors = load_file('model\model.safetensors')

# Model path
model_path = 'model'

# Load the model
model = SentenceTransformer(model_path)

model.set_weights(tensors)

# Save the model in .h5 format
model.save('model.h5')