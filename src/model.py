import copy
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D, Downsample2D, Upsample2D
from diffusers.models.unet_2d_blocks import get_down_block
from diffusers.models.unet_2d_condition import (
    UNet2DConditionOutput,
    UNet2DConditionModel,
)
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


