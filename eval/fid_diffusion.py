from torchvision.transforms import functional as F
from PIL import Image
import os
import numpy as np
import torch
from diffusers import DiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusers.utils.logging import tqdm

dataset_path = "./102flowers"
model_id = "./ddpm-ema-flowers-256"
num_inference_steps = 1000
batch_size = 16
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, [256, 256])


print("Loading real images...")

image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])
real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
real_images = torch.cat([preprocess_image(image) for image in real_images])
print("Finished loading real images. Real images shape:", real_images.shape)

print("\n***********************************************")
print("Loading model...")
pipeline = DiffusionPipeline.from_pretrained(model_id).to(device)

print("Start generating images...")
fake_images = torch.tensor([]).to(device)

for _ in tqdm(range(0, len(real_images), batch_size)):
    current_batch_size = min(batch_size, len(real_images) - len(fake_images))
    batch_images = pipeline(batch_size=current_batch_size, num_inference_steps=num_inference_steps).images
    batch_images = torch.tensor([np.array(img) for img in batch_images]) / 255.0
    fake_images = torch.cat([fake_images, batch_images.permute(0, 3, 1, 2).to(device)])

print("Images generation end. Shape:", fake_images.shape)
print("***********************************************\n")
print("Start evaluating FID...")

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
