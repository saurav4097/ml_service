from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from predict import predict_image
import io

app = FastAPI()

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
