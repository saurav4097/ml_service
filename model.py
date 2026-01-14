from transformers import CLIPProcessor, CLIPModel
MODEL_NAME = "openai/clip-vit-base-patch32"

model = None
processor = None
def load_model():
    global model, processor
    if model is None or processor is None:
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model.eval()
