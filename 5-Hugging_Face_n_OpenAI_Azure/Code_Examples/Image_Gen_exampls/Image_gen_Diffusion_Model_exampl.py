import torch
from diffusers import StableDiffusionPipeline

#model_id = "dreamlike-art/dreamlike-photoreal-2.0"
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

positive_prompts = [
    "A serene sunset over a calm lake",
    "A bustling cityscape at night",
    "A tranquil forest with sunlight filtering through the trees"
]

negative_prompts = [
    "blurry, distorted, low quality",
    "bad lighting, messy composition, low resolution",
    "dark, unclear, out of focus"
]

generated_images = []

for i, (prompt, neg_prompt) in enumerate(zip(positive_prompts, negative_prompts)):
    image = pipeline(prompt=prompt, negative_prompt=neg_prompt).images[0]
    image.save(f'image_{i}.jpg')
    generated_images.append(image)

len(generated_images)


