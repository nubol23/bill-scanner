from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageOps

from .base import InvalidImageError


@dataclass(frozen=True)
class PreparedImage:
  name: str
  image: Image.Image


def load_image_bytes(image_bytes: bytes) -> Image.Image:
  if not image_bytes:
    raise InvalidImageError("The uploaded image is empty.")

  try:
    image = Image.open(BytesIO(image_bytes))
    image.load()
  except Exception as error:
    raise InvalidImageError("The uploaded file is not a valid image.") from error

  return ImageOps.exif_transpose(image).convert("RGB")


def build_candidate_images(image: Image.Image, *, top_crop_ratio: float) -> list[PreparedImage]:
  width, height = image.size
  candidates = [PreparedImage(name="full", image=image)]

  if height >= 40 and width / max(height, 1) >= 1.75:
    top_height = max(1, min(height, int(math.ceil(height * top_crop_ratio))))
    candidates.append(
      PreparedImage(
        name="top-band",
        image=image.crop((0, 0, width, top_height)),
      )
    )

  return candidates


def prepare_onnx_input(
  image: Image.Image,
  *,
  image_height: int,
  min_width: int,
  channel_order: str,
  mean: tuple[float, float, float],
  std: tuple[float, float, float],
  pad_value: float,
) -> np.ndarray:
  width, height = image.size
  if width <= 0 or height <= 0:
    raise InvalidImageError("The uploaded image has invalid dimensions.")

  aspect_ratio = width / float(height)
  resized_width = max(1, int(math.ceil(image_height * aspect_ratio)))
  padded_width = max(min_width, resized_width)

  resized_image = image.resize((resized_width, image_height), Image.Resampling.BILINEAR)
  array = np.asarray(resized_image, dtype=np.float32)

  if channel_order.upper() == "BGR":
    array = array[:, :, ::-1]

  array /= 255.0
  array = (array - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
  array = array.transpose((2, 0, 1))

  padded = np.full((1, 3, image_height, padded_width), pad_value, dtype=np.float32)
  padded[0, :, :, :resized_width] = array
  return padded
