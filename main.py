# 📁 main.py
# ----------------------------------------------------
# FastAPI app for serving a fine-tuned DistilBERT model
# Includes sentiment prediction API with tokenizer + model loaded from local files
# All code comments are based on your earlier questions
# ----------------------------------------------------

import os
import requests
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch


MODEL_URL = "https://huggingface.co/datasets/Muskan2023/sentiment-model/resolve/main/sentiment_model.pt"
MODEL_PATH = "model/sentiment_model.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("📥 Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ⬇️ Happens on startup
    download_model()
    yield
    # ⬇️ Cleanup logic here (on shutdown), if needed

# 🔹 FastAPI app instance
app = FastAPI(lifespan=lifespan)

# ❗️Use CORS middleware only if your frontend and backend are on different origins (e.g., different ports/domains).
# 👉 Example: frontend on http://localhost:3000 and backend on http://localhost:8000
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins = ["http://0.0.0.0:8080"],
#     allow_credentials = True,
#     allow_methods = ["*"],
#     allow_headers = ["*"]
# )

# ✅ No need for CORS if you're serving frontend (index.html, JS) and backend API from the same FastAPI app.
# 👉 StaticFiles serves HTML/JS from the same origin as API, so browser allows fetch requests without CORS.
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def redirect_to_static():
    return RedirectResponse(url="/static/")

class InputText(BaseModel):
    text: str

# 🔹 Load tokenizer (saved locally after training, ensures consistency)
# Instead of loading from HuggingFace hub, this loads the exact one used in training
# Which ensures consistent tokenization, and avoids future mismatch due to library changes
tokenizer = DistilBertTokenizerFast.from_pretrained("./tokenizer")

# 🔹 Load trained model from local .pt file
# .pt is the standard PyTorch model checkpoint file extension
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("./model/sentiment_model.pt", map_location=torch.device('cpu')))

# 🔹 Set model to evaluation mode to disable dropout etc., for consistent inference results
model.eval()

# 🔹 Inference route
@app.post("/predict")
def predict(input: InputText):
    # Tokenize input (same method used in training)
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    # 🔹 Inference under no_grad context to save memory and speed up
    # Gradient tracking is unnecessary and wasteful during prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # 🔹 Extract logits (raw scores before softmax)
    # Higher score = model is more confident in that class
    logits = outputs.logits

    # 🔹 Convert logits to prediction (class index with highest score)
    # torch.argmax returns the index of the highest logit → final class prediction (0 or 1)
    prediction = torch.argmax(logits, dim=1).item()

    # 🔹 Optional: You can also return probability using softmax if needed

    return {"prediction": prediction}  # 0 or 1
