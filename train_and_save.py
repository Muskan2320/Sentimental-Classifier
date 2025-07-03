# --------------------------------------------------------
# This script:
# 1. Loads IMDB dataset
# 2. Tokenizes text using DistilBERT tokenizer
# 3. Fine-tunes DistilBERTForSequenceClassification
# 4. Saves model and tokenizer locally for inference use
# --------------------------------------------------------

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# 🔹 Load IMDB dataset from Hugging Face Datasets
dataset = load_dataset("imdb")

# 🔹 Load tokenizer (distilbert-base-uncased = lowercase only)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 🔹 Tokenize function — truncates/pads to 256 tokens (most reviews < 256 tokens)
# Why 256? ~85% IMDB reviews are under 256 tokens; 512 is costly for compute and memory
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

# 🔹 Apply tokenization to the train/test datasets
tokenized = dataset.map(tokenize_function, batched=True)

# 🔹 Choose a subset of training and test data to speed up fine-tuning
# Full IMDB = 25K samples. Here we use 10K train and 1K test for faster training
train_dataset = tokenized["train"].shuffle(seed=42).select(range(10000))
eval_dataset  = tokenized["test"].select(range(1000))

# 🔹 Load pre-trained DistilBERT model for classification
# `num_labels=2` → binary classification (positive/negative)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 🔹 Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Where to save checkpoints
    num_train_epochs=1,              # Train for 1 epoch for demo speed
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
)

# 🔹 Use Hugging Face Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 🔹 Train the model
trainer.train()

# 🔹 Save the model (state dict only — smaller)
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/sentiment_model.pt")  # .pt = PyTorch checkpoint

# 🔹 Save the tokenizer (must be saved for exact tokenization during inference)
tokenizer.save_pretrained("tokenizer")
