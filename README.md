## Training an unconditional latent diffusion model

Creating a training image set is [described in a different document](https://huggingface.co/docs/datasets/image_process#image-datasets).

### Cloning to local
```bash
git clone https://github.com/zyinghua/uncond-image-generation-ldm.git
```

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install diffusers
pip install peft
```

Then navigate to the uncond-image-generation-ldm folder containing the training script and install the required dependencies:
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Unconditional Flowers

The command to train a DDPM UNet model on the Oxford Flowers dataset:

```bash
accelerate launch train.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=256 \
  --output_dir="ddpm-ema-flowers-256" \
  --train_batch_size=16 \
  --num_epochs=150 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```

### Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. After setting up with `accelerate config`,
simply add `--multi_gpu` in the command. For more information, follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash
accelerate launch --multi_gpu train.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=256 \
  --output_dir="ddpm-ema-flowers-256" \
  --train_batch_size=16 \
  --num_epochs=150 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```

To be able to use Weights and Biases (`wandb`) as a logger you need to install the library: `pip install wandb`.

### Using your own data

To use your own dataset, there are 3 ways:
- you can either provide your own folder as `--train_data_dir`
- or you can provide your own .zip file containing the data as `--train_data_files`
- or you can upload your dataset to the hub (possibly as a private repo, if you prefer so), and simply pass the `--dataset_name` argument.

Below, we explain both in more detail.

#### Provide the dataset as a folder

If you provide your own folders with images, the script expects the following directory structure:

```bash
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```

In other words, the script will take care of gathering all images inside the folder. You can then run the script like this:

```bash
accelerate launch train.py \
    --train_data_dir <path-to-train-directory> \
    <other-arguments>
```

Internally, the script will use the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature which will automatically turn the folders into 🤗 Dataset objects.

#### Upload your data to the hub, as a (possibly private) repo

It's very easy (and convenient) to upload your image dataset to the hub using the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature available in 🤗 Datasets. Simply do the following:

```python
from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_your_folder")

# example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path_to_zip_file")

# example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip")

# example 4: providing several splits
dataset = load_dataset("imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]})
```

`ImageFolder` will create an `image` column containing the PIL-encoded images.

Next, push it to the hub!

```python
# assuming you have ran the huggingface-cli login command in a terminal
dataset.push_to_hub("name_of_your_dataset")

# if you want to push to a private repo, simply pass private=True:
dataset.push_to_hub("name_of_your_dataset", private=True)
```

and that's it! You can now train your model by simply setting the `--dataset_name` argument to the name of your dataset on the hub.

More on this can also be found in [this blog post](https://huggingface.co/blog/image-search-datasets).
