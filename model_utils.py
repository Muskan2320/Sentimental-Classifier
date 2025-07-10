# ----------------------------------------------------
# Utilities for downloading and loading the trained sentiment model (.pt)
# ----------------------------------------------------

import os
import requests
import torch
from transformers import DistilBertForSequenceClassification

# 🔹 Direct model download link (hosted on Hugging Face datasets repo)
MODEL_URL = "https://huggingface.co/datasets/Muskan2023/sentiment-model/resolve/main/sentiment_model.pt"
MODEL_PATH = "model/sentiment_model.pt"

# 🔹 Download the model only if not already available locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("📥 Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded.")

# Load model from .pt file and prepare for inference
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()  # set model to evaluation mode 
    return model