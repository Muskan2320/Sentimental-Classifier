# 📊 Sentiment Classifier — DistilBERT + FastAPI + Docker + UI

This project demonstrates a full AI workflow: training a DistilBERT model on IMDB movie reviews, serving it via a FastAPI backend, containerizing it with Docker, and testing it with a simple UI.

---

## ✅ Features

* Fine-tunes DistilBERT on IMDB for binary sentiment classification (positive/negative)
* Tokenizes reviews using Hugging Face Tokenizer
* Trains model using Hugging Face Trainer
* Saves model (.pt) and tokenizer locally
* Serves predictions via FastAPI
* Dockerized for production-ready deployment
* Includes simple browser-based UI to test the API
 
---

## 🧠 Tech Stack

* 🤗 Hugging Face Transformers + Datasets
* 🧠 DistilBERT (pretrained model)
* 🔥 PyTorch
* ⚡ FastAPI
* 🐳 Docker
* 🌐 HTML + JavaScript (for UI)

---

## 📁 Project Structure

```
├── tokenizer/              # Saved tokenizer files
├── main.py                 # FastAPI API for inference
├── train_and_save.py       # Script to train and save model/tokenizer
├── requirements.txt
├── Dockerfile
├── index.html              # Simple browser UI
└── README.md
```

---

## 🏁 How to Run (Locally)

### 🔹 1. Train the model

```bash
python train_and_save.py
```

* Downloads IMDB dataset
* Fine-tunes DistilBERT for 1 epoch on 10k samples
* Saves `sentiment_model.pt` and tokenizer

### 🔹 2. Serve with FastAPI

```bash
uvicorn main:app --reload
```

* FastAPI server runs on `http://localhost:8000`
* Test at `http://localhost:8000/docs`

### 🔹 3. Test with UI

Open `index.html` in a browser and enter a movie review.

---

## 🐳 Docker Instructions

### 🔹 Build Docker Image

```bash
docker build -t sentiment-classifier .
```

### 🔹 Run Container

```bash
docker run -p 8000:8000 sentiment-classifier
```

### 🔹 Access FastAPI

Go to `http://localhost:8000/docs` to test Swagger UI.

---

## 🔍 Notes

* `256` max token length used instead of 512 — covers \~85% of reviews efficiently
* Model trained on subset (10k train / 1k eval) for fast prototyping
* Full training on 25k samples improves accuracy
* Tokenizer is saved locally to ensure identical preprocessing at inference

---

## 🎯 Prediction Output

Returns `{ "prediction": 1 }` → where 1 = Positive, 0 = Negative

---
