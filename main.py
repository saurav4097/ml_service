from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from predict import predict_image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ✅ Call with single argument
        result = predict_image(image)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))