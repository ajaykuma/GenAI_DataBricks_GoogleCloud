import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# -----------------------------
# Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion
#model_id = "dreamlike-art/dreamlike-photoreal-2.0"
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

os.makedirs("final_outputs", exist_ok=True)

# -----------------------------
# CLIP Scoring Functions
# -----------------------------
def get_clip_score(image: Image.Image, text: str) -> float:
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = outputs.logits_per_text.softmax(dim=1).squeeze().item()
    return score

def contrastive_score(image: Image.Image, positive: str, negative: str) -> float:
    pos_score = get_clip_score(image, positive)
    neg_score = get_clip_score(image, negative)
    return pos_score - neg_score  # TRUE contrastive scoring


# -----------------------------
# Main Pipeline Function
# -----------------------------
def generate_and_rank(prompt, contrast_prompt=None, num_samples=5):
    images = []

    # Step 1: Generate multiple samples
    for _ in range(num_samples):
        img = pipeline(prompt=prompt).images[0]
        images.append(img)

    scored_images = []

    # Step 2: Score images
    for img in images:
        if contrast_prompt:
            score = contrastive_score(img, prompt, contrast_prompt)
        else:
            score = get_clip_score(img, prompt)

        scored_images.append((img, score))

    # Step 3: Rank images
    ranked = sorted(scored_images, key=lambda x: -x[1])

    return ranked


# -----------------------------
# Example Usage (Contrastive Pairs)
# -----------------------------
prompt_pairs = [
    ("A man with a beard", "A man without a beard"),
    ("A bright sunny beach", "A dark stormy beach"),
    ("A cat sitting on a sofa", "A dog sitting on a sofa"),
]

for i, (pos_prompt, neg_prompt) in enumerate(prompt_pairs):
    print(f"\nProcessing: {pos_prompt} vs {neg_prompt}")

    ranked_images = generate_and_rank(
        prompt=pos_prompt,
        contrast_prompt=neg_prompt,
        num_samples=5
    )

    # Best image
    best_img, best_score = ranked_images[0]

    # Save best
    best_img.save(f"final_outputs/best_{i}.jpg")

    print(f"Best Score: {best_score:.4f}")

    # Optional: Save all for inspection
    for j, (img, score) in enumerate(ranked_images):
        img.save(f"final_outputs/{i}_{j}_{score:.4f}.jpg")
