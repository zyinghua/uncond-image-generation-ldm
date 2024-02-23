import torch
from diffusers import DiffusionPipeline
from diffusers.utils.logging import tqdm
import numpy as np
from PIL import Image
import os

model_id = "../../ddpm-ema-flowers-256"
output_dir = "./102flowers-generated"
num_inference_steps = 1000
num_images = 16
per_batch_size = 16
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if not os.path.exists(output_dir):
    """If output_dir does not exist, create one."""
    os.makedirs(output_dir, exist_ok=True)

print("Loading model...")
pipeline = DiffusionPipeline.from_pretrained(model_id).to(device)

print("Start generating images...")
fake_images = torch.tensor([]).to(device)

for _ in tqdm(range(0, num_images, per_batch_size)):
    current_batch_size = min(per_batch_size, num_images - len(fake_images))
    batch_images = pipeline(batch_size=current_batch_size, num_inference_steps=num_inference_steps).images
    batch_images = torch.tensor([np.array(img) for img in batch_images])
    fake_images = torch.cat([fake_images, batch_images.to(device)])

print("Images generation end. Shape:", fake_images.shape)
print("***********************************************\n")

for i, img in enumerate(fake_images):
    img_pil = Image.fromarray(img.cpu().numpy().astype("uint8"))
    img_pil.save(output_dir + f"/image_{i}.png")
