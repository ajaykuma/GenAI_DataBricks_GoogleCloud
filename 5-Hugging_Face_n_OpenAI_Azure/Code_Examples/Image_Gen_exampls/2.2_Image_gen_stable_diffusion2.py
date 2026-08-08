import torch
from diffusers import StableDiffusionPipeline

#model_id = "dreamlike-art/dreamlike-photoreal-2.0"
model_id = "runwayml/stable-diffusion-v1-5"

#Online and if GPU
#pipeline = StableDiffusionPipeline.from_pretrained(model_id)

#Offline and with CPU
# Load model completely offline
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    local_files_only=True,
    dtype=torch.float32
)

# Use CPU
pipeline = pipeline.to("cpu")

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
    #when GPU
    #image = pipeline(prompt=prompt, negative_prompt=neg_prompt).images[0]

    #when cpu
    image = pipeline(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=15,
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]

    image.save(f'image_{i}.jpg')
    generated_images.append(image)

print("Generated images:", len(generated_images))


