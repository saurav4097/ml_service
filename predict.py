from PIL import Image
import torch
import random
from model import model, processor

# 🔒 Fixed outer-appearance personality labels
LABELS = [
    "Confidence",
    "Dominance",
    "Attractiveness",
    "Style",
    "Clarity",
    "Sharpness",
    "Attitude"
]

def humanize_score(x: torch.Tensor, min_out=50, max_out=95, gamma=0.55):
    """
    Converts CLIP relative score → human-friendly percentage
    - Ensures minimum confidence (>=50)
    - Boosts mid/low values
    - Keeps realism (not all 100s)
    """
    x = torch.clamp(x, 0.0, 1.0)
    x = x ** gamma
    return min_out + x * (max_out - min_out)

def predict_image(image: Image.Image):
    if image is None:
        raise ValueError("Invalid image")
    
    # Ensure model runs on CPU (safe for Railway / free hosting)
    device = torch.device("cpu")
    model.to(device)

    # CLIP input
    inputs = processor(
        text=LABELS,
        images=image,
        return_tensors="pt",
        padding=True
    )

  # Move tensors to CPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Relative probabilities (ranking, not truth)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

    # Normalize by strongest trait
    max_prob = probs.max()
    normalized = probs / max_prob

    scores = {}

    for i, label in enumerate(LABELS):
        base_score = humanize_score(normalized[i])

        # 🎯 tiny randomness → natural look
        jitter = random.uniform(-2.0, 2.0)
        final_score = torch.clamp(base_score + jitter, 50.0, 95.0)

        scores[label] = round(float(final_score), 1)

    return {
        "scores": scores
    }