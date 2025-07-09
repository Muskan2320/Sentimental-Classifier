# 🔹 Start from a lightweight Python 3.10 image
FROM python:3.10-slim

# 🔹 Set working directory inside container
WORKDIR /app

# 🔹 Copy requirements and app code into the container
COPY requirements.txt .
COPY . .

# 🔹 Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Start FastAPI app using uvicorn (host 0.0.0.0 = accessible externally)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]