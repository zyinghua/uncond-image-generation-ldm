"""Generate cvd images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch
import pickle
from diffusers import DiffusionPipeline

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--pipeline', 'pipeline_path', help='Directory path of the pipeline to load', required=True)
@click.option('--num_inference_steps', 'num_inference_steps', help='Number of inference steps', default=1000)
@click.option('--degree', 'degree', type=float, default=0.0, help='CVD degree')
@click.option('--num_images', 'num_images', type=int, help='Total number of images to generate', default=480)
@click.option('--grid_rows', 'grid_rows', type=int, help='Number of image grid rows', default=16)
@click.option('--grid_cols', 'grid_cols', type=int, help='Number of image grid columns', default=30)
@click.option('--outdir', 'outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    pipeline_path: str,
    num_inference_steps: int,
    degree: float,
    num_images: int,
    grid_rows: int,
    grid_cols: int,
    outdir: str,
):
    """Sanity check for the inputs"""

    if grid_rows * grid_cols != num_images:
        ctx.fail('Number of rows and columns must multiply to the total number of images')

    if degree < 0.0 or degree > 1.0:
        ctx.fail('CVD degree must be within the range [0, 1]')

    """Generate images using pretrained pipeline."""
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print('Loading pipeline from "%s"...' % pipeline_path)
    pipeline = DiffusionPipeline.from_pretrained(pipeline_path).to(device)

    if not os.path.exists(outdir):
        """If outdir does not exist, create one."""
        os.makedirs(outdir, exist_ok=True)

    generated_images = torch.tensor([]).to(device)

    for r in range(grid_rows):
        images = pipeline(batch_size=grid_cols, num_inference_steps=num_inference_steps).images # (N, H, W, C)
        image_row = torch.cat([torch.tensor(np.array(img)) for img in images], dim=1).to(device)

        generated_images = torch.cat([generated_images, image_row], dim=0)

    generated_images = generated_images.clamp(0, 255).to(torch.uint8)

    PIL.Image.fromarray(generated_images.cpu().numpy(), 'RGB').save(os.path.join(outdir, 'generated_images.png'))



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()

#----------------------------------------------------------------------------
