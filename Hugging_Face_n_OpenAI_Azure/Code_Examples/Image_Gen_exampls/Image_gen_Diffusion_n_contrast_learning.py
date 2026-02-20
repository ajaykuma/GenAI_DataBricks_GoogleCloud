from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def rank_images_by_clip(prompt, images):
    inputs = clip_processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    scores = outputs.logits_per_text.softmax(dim=1).squeeze()
    return sorted(zip(images, scores), key=lambda x: -x[1])  # Highest score first

#####
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Load Stable Diffusion
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Pairs of prompts with contrasting semantics
prompt_pairs = [
    ("A man with a beard", "A man without a beard"),
    ("A bright sunny beach", "A dark stormy beach"),
    ("A cat sitting on a sofa", "A dog sitting on a sofa"),
]

os.makedirs("contrastive_outputs", exist_ok=True)

def get_clip_similarity(image: Image.Image, text: str) -> float:
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarity = outputs.logits_per_text.softmax(dim=1).squeeze().item()
    return similarity

for i, (positive_prompt, contrast_prompt) in enumerate(prompt_pairs):
    # Generate images
    pos_img = pipeline(prompt=positive_prompt).images[0]
    neg_img = pipeline(prompt=contrast_prompt).images[0]

    # Save images
    pos_img.save(f"contrastive_outputs/pos_{i}.jpg")
    neg_img.save(f"contrastive_outputs/neg_{i}.jpg")

    # CLIP similarity
    sim_pos = get_clip_similarity(pos_img, positive_prompt)
    sim_neg = get_clip_similarity(neg_img, contrast_prompt)

    print(f"\nPrompt Pair {i+1}:")
    print(f"  {positive_prompt} -> CLIP Score: {sim_pos:.4f}")
    print(f"  {contrast_prompt} -> CLIP Score: {sim_neg:.4f}")
