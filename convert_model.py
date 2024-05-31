import torch
from sentence_transformers import SentenceTransformer

# Model path
model_path = 'model'

# Load the model
model = SentenceTransformer(model_path)

# Set the device to CPU or CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Dummy input for tracing
dummy_input = torch.randint(0, 100, (1, 512), dtype=torch.long).to(device)  # Make sure dummy_input is on the same device

# Convert to ONNX
torch.onnx.export(model._first_module().auto_model, dummy_input, "model.onnx")