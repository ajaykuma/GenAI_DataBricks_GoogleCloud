##GPU Version
# import torch
# from diffusers import StableDiffusionPipeline
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import os

# # -----------------------------
# # Setup
# # -----------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load Stable Diffusion
# #model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# model_id = "runwayml/stable-diffusion-v1-5"
# pipeline = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32
# ).to(device)

# # Load CLIP
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# os.makedirs("final_outputs", exist_ok=True)

# # -----------------------------
# # CLIP Scoring Functions
# # -----------------------------
# def get_clip_score(image: Image.Image, text: str) -> float:
#     inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         outputs = clip_model(**inputs)
#         score = outputs.logits_per_text.softmax(dim=1).squeeze().item()
#     return score

# def contrastive_score(image: Image.Image, positive: str, negative: str) -> float:
#     pos_score = get_clip_score(image, positive)
#     neg_score = get_clip_score(image, negative)
#     return pos_score - neg_score  # TRUE contrastive scoring


# # -----------------------------
# # Main Pipeline Function
# # -----------------------------
# def generate_and_rank(prompt, contrast_prompt=None, num_samples=5):
#     images = []

#     # Step 1: Generate multiple samples
#     for _ in range(num_samples):
#         img = pipeline(prompt=prompt).images[0]
#         images.append(img)

#     scored_images = []

#     # Step 2: Score images
#     for img in images:
#         if contrast_prompt:
#             score = contrastive_score(img, prompt, contrast_prompt)
#         else:
#             score = get_clip_score(img, prompt)

#         scored_images.append((img, score))

#     # Step 3: Rank images
#     ranked = sorted(scored_images, key=lambda x: -x[1])

#     return ranked


# # -----------------------------
# # Example Usage (Contrastive Pairs)
# # -----------------------------
# prompt_pairs = [
#     ("A man with a beard", "A man without a beard"),
#     ("A bright sunny beach", "A dark stormy beach"),
#     ("A cat sitting on a sofa", "A dog sitting on a sofa"),
# ]

# for i, (pos_prompt, neg_prompt) in enumerate(prompt_pairs):
#     print(f"\nProcessing: {pos_prompt} vs {neg_prompt}")

#     ranked_images = generate_and_rank(
#         prompt=pos_prompt,
#         contrast_prompt=neg_prompt,
#         num_samples=5
#     )

#     # Best image
#     best_img, best_score = ranked_images[0]

#     # Save best
#     best_img.save(f"final_outputs/best_{i}.jpg")

#     print(f"Best Score: {best_score:.4f}")

#     # Optional: Save all for inspection
#     for j, (img, score) in enumerate(ranked_images):
#         img.save(f"final_outputs/{i}_{j}_{score:.4f}.jpg")

##CPU Version
import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


# ============================================================
# SETUP
# ============================================================

device = "cpu"

print("Using device:", device)


# ============================================================
# LOAD STABLE DIFFUSION - OFFLINE
# ============================================================

model_id = "runwayml/stable-diffusion-v1-5"

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    local_files_only=True,
    dtype=torch.float32
)

pipeline = pipeline.to(device)

print("Stable Diffusion loaded.")


# ============================================================
# LOAD CLIP - OFFLINE
# ============================================================

clip_model_id = "openai/clip-vit-base-patch32"

clip_model = CLIPModel.from_pretrained(
    clip_model_id,
    local_files_only=True
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    clip_model_id,
    local_files_only=True
)

print("CLIP loaded.")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "final_outputs",
    exist_ok=True
)


# ============================================================
# CLIP SCORING
# ============================================================

def get_clip_score(image: Image.Image, text: str) -> float:

    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Move tensors to CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }

    with torch.no_grad():

        outputs = clip_model(**inputs)

    # Raw CLIP image-text similarity
    score = outputs.logits_per_image[0, 0].item()

    return score


# ============================================================
# CONTRASTIVE CLIP SCORE
# ============================================================

def contrastive_score(
    image: Image.Image,
    positive: str,
    negative: str
) -> float:

    inputs = clip_processor(
        text=[positive, negative],
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Move tensors to CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }

    with torch.no_grad():

        outputs = clip_model(**inputs)

    # Similarity of image against both prompts
    logits = outputs.logits_per_image[0]

    positive_score = logits[0].item()
    negative_score = logits[1].item()

    # TRUE contrastive score
    return positive_score - negative_score


# ============================================================
# GENERATE AND RANK
# ============================================================

def generate_and_rank(
    prompt,
    contrast_prompt=None,
    num_samples=5
):

    images = []

    # --------------------------------------------------------
    # Step 1: Generate multiple images
    # --------------------------------------------------------

    print(f"Generating {num_samples} images...")

    for sample_number in range(num_samples):

        print(
            f"  Generating image "
            f"{sample_number + 1}/{num_samples}"
        )

        img = pipeline(
            prompt=prompt,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]

        images.append(img)

    # --------------------------------------------------------
    # Step 2: Score images
    # --------------------------------------------------------

    scored_images = []

    print("Scoring images with CLIP...")

    for img in images:

        if contrast_prompt:

            score = contrastive_score(
                img,
                prompt,
                contrast_prompt
            )

        else:

            score = get_clip_score(
                img,
                prompt
            )

        scored_images.append(
            (img, score)
        )

    # --------------------------------------------------------
    # Step 3: Rank images
    # --------------------------------------------------------

    ranked = sorted(
        scored_images,
        key=lambda x: -x[1]
    )

    return ranked


# ============================================================
# CONTRASTIVE PROMPT PAIRS
# ============================================================

prompt_pairs = [

    (
        "A man with a beard",
        "A man without a beard"
    ),

    (
        "A bright sunny beach",
        "A dark stormy beach"
    ),

    (
        "A cat sitting on a sofa",
        "A dog sitting on a sofa"
    )
]


# ============================================================
# RUN EXPERIMENT
# ============================================================

for i, (pos_prompt, neg_prompt) in enumerate(prompt_pairs):

    print("\n" + "=" * 60)

    print(
        f"Processing:\n"
        f"  Positive: {pos_prompt}\n"
        f"  Negative: {neg_prompt}"
    )

    print("=" * 60)

    ranked_images = generate_and_rank(
        prompt=pos_prompt,
        contrast_prompt=neg_prompt,
        num_samples=5
    )

    # --------------------------------------------------------
    # Best image
    # --------------------------------------------------------

    best_img, best_score = ranked_images[0]

    best_img.save(
        f"final_outputs/best_{i}.jpg"
    )

    print(
        f"\nBest contrastive score: "
        f"{best_score:.4f}"
    )

    # --------------------------------------------------------
    # Save all ranked images
    # --------------------------------------------------------

    for j, (img, score) in enumerate(ranked_images):

        img.save(
            f"final_outputs/"
            f"{i}_{j}_{score:.4f}.jpg"
        )

    print("Images saved.")
