from torchvision.transforms import functional as F
from PIL import Image
import os
import numpy as np
import torch
from diffusers import DiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusers.utils.logging import tqdm

real_dataset_path = "./102flowers-processed256"
fake_dataset_path = "./102flowers-generated"

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

real_image_paths = sorted([os.path.join(real_dataset_path, x) for x in os.listdir(real_dataset_path)])
real_images = [np.array(Image.open(path).convert("RGB")) for path in real_image_paths]
real_images = torch.cat([preprocess_image(image) for image in real_images])
print("Finished loading real images. Real images shape:", real_images.shape)

print("\n***********************************************")
print("Loading fake images...")

fake_image_paths = sorted([os.path.join(fake_dataset_path, x) for x in os.listdir(fake_dataset_path)])
fake_images = [np.array(Image.open(path).convert("RGB")) for path in fake_image_paths]
fake_images = torch.cat([preprocess_image(image) for image in fake_images])
print("Finished loading fake images. Fake images shape:", fake_images.shape)

print("Start evaluating FID...")

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
