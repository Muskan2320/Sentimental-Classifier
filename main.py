from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from model_utils import download_model, load_model
from predict_utils import predict

# 🔹 FastAPI model structure
class InputText(BaseModel):
    text: str

# 🔹 Lifespan startup logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    download_model()
    global model
    model = load_model()
    yield

# 🔹 FastAPI app
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
# 🔹 Static frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def redirect_to_static():
    return RedirectResponse(url="/static/")

# 🔹 API prediction endpoint
@app.post("/predict")
def predict_sentiment(input: InputText):
    result = predict(model, input.text)
    return {"prediction": result}