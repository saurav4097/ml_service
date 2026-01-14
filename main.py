from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from predict import predict_image
import io

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Personality labels
DEFAULT_LABELS = [
    "Attractiveness",
    "Confidence",
    "Dominance",
    "Style",
    "Sharpness",
    "Clarity",
    "Attitude"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict_image(image, DEFAULT_LABELS)
    return result
