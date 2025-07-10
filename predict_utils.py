# ----------------------------------------------------
# Tokenizer + prediction logic for sentiment classifier
# ----------------------------------------------------

import torch
from transformers import DistilBertTokenizerFast

# Load tokenizer (saved locally after training, ensures consistency)
# Instead of loading from HuggingFace hub, this loads the exact one used in training
# Which ensures consistent tokenization, and avoids future mismatch due to library changes
tokenizer = DistilBertTokenizerFast.from_pretrained("./tokenizer")

# Prediction function
def predict(model, text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    # Inference under no_grad context to save memory and speed up
    # Gradient tracking is unnecessary and wasteful during prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # logits are raw scores before softmax
    logits = outputs.logits

    # Convert logits to prediction (class index with highest score)
    prediction = torch.argmax(logits, dim=1).item()

    return prediction  # 0 or 1