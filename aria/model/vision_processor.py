# Copyright 2024 Rhymes AI. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import BaseImageProcessor, BatchFeature, TensorType


def _select_best_resolution(
    img_width: int, img_height: int, target_ratios: List[List[int]], patch_size: int
):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        img_width: the original widths of images.
        img_height: the original heights of images.
        target_ratios (2d numpy array): dimension size (M,2)
        patch_size (int): image patch size

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """

    aspect_ratio = img_width / img_height
    best_ratio_diff = float("inf")
    best_ratio_w, best_ratio_h = 1, 1
    area = np.int32(img_height) * np.int32(img_height)
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio_w, best_ratio_h = ratio[0], ratio[1]
        elif (
            ratio_diff == best_ratio_diff
            and area > 0.5 * patch_size * patch_size * ratio[0] * ratio[1]
        ):
            best_ratio_w, best_ratio_h = ratio[0], ratio[1]

    return best_ratio_w, best_ratio_h


def _split_image(
    image: Image.Image,
    split_image: bool,
    split_ratio: List[List[int]],
    patch_size: int,
) -> List[Image.Image]:
    """
    Split image into multiple patches

    Args:
        image (PIL.Image): Input image.
        split_image (bool): Whether to split the image into patches.
        split_ratio (2d numpy array): dimension size (M,2)
        patch_size (int): image patch size

    Returns:
        List[PIL.Image]: List of splitted images.
    """
    if split_image:
        ratio_width, ratio_height = _select_best_resolution(
            image.width, image.height, split_ratio, patch_size
        )
        resize_width = patch_size * ratio_width
        resize_height = patch_size * ratio_height
        blocks = ratio_width * ratio_height
        resized_img = image.resize((resize_width, resize_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (resize_width // patch_size)) * patch_size,
                (i // (resize_width // patch_size)) * patch_size,
                ((i % (resize_width // patch_size)) + 1) * patch_size,
                ((i // (resize_width // patch_size)) + 1) * patch_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if len(processed_images) != 1:
            processed_images.insert(0, image)
        return processed_images
    else:
        return [image]


def keep_ratio_resize_and_pixel_mask(
    img: Image.Image, max_size, min_size=336, padding_value=0
):
    """
    Resize an image while maintaining aspect ratio and create a pixel mask.

    Args:
        img (PIL.Image): Input image.
        max_size (int): Maximum size for the larger dimension of the image.
        min_size (int, optional): Minimum size for the smaller dimension. Defaults to 336.
        padding_value (int, optional): Value used for padding. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - PIL.Image: Resized and padded image.
            - torch.Tensor: Boolean pixel mask. This mask is a 2D tensor of shape (max_size, max_size) where:
                - True (1) values indicate pixels that belong to the original resized image.
                - False (0) values indicate pixels that are part of the padding.
              The mask helps distinguish between actual image content and padded areas in subsequent processing steps.
    """
    img = img.convert("RGB")
    # rescale the given image, keep the aspect ratio
    scale = max_size / max(img.size)

    w, h = img.size
    if w >= h:
        new_size = (max_size, max(int(h * scale), min_size))  # w, h
    else:
        new_size = (max(int(w * scale), min_size), max_size)  # w, h

    img_resized = img.resize(new_size, resample=Image.Resampling.BICUBIC)

    # padding the right/bottom
    padding_right, padding_bottom = max_size - new_size[0], max_size - new_size[1]
    img_padded = ImageOps.expand(
        img_resized, (0, 0, padding_right, padding_bottom), fill=padding_value
    )

    # Create a pixel mask
    pixel_mask = torch.zeros(max_size, max_size)
    pixel_mask[: new_size[1], : new_size[0]] = 1
    pixel_mask = pixel_mask.bool()
    return img_padded, pixel_mask


class AriaVisionProcessor(BaseImageProcessor):
    """
    A vision processor for the Aria model that handles image preprocessing.
    """

    def __init__(
        self,
        max_image_size=980,
        min_image_size=336,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        **kwargs,
    ):
        """
        Initialize the AriaVisionProcessor.

        Args:
            max_image_size (int, optional): Maximum image size. Defaults to 980.
            min_image_size (int, optional): Minimum image size. Defaults to 336.
            mean (list, optional): Mean values for normalization. Defaults to [0.5, 0.5, 0.5].
            std (list, optional): Standard deviation values for normalization. Defaults to [0.5, 0.5, 0.5].
        """
        super().__init__(**kwargs)

        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.auto_map = {
            "AutoProcessor": "processing_aria.AriaProcessor",
            "AutoImageProcessor": "vision_processor.AriaVisionProcessor",
        }

        # we make the transform a property so that it is lazily initialized,
        # this could avoid the error "TypeError: Object of type Normalize is not JSON serializable"
        # when we used save_pretrained or from_pretrained.
        self._transform = None
        self._set_processor_class("AriaProcessor")

    @property
    def transform(self):
        if self._transform is None:
            # Recreate the transform when accessed
            self._transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.image_mean, self.image_std),
                ]
            )
        return self._transform

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        max_image_size: Optional[int] = 980,
        min_image_size: Optional[int] = 336,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        split_image: Optional[bool] = False,
        split_ratio: Optional[List[List[int]]] = [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 2],
            [2, 1],
            [3, 1],
            [4, 1],
        ],
    ):
        """
        Process a list of images.

        Args:
            images (list): List of PIL.Image objects.
            max_image_size (int, optional): Override the default max image size. Defaults to None.
            return_tensors (str or TensorType, optional): The type of tensor to return. Defaults to "pt".
            split_image (bool, optional): Whether to split the image. Defaults to False.
            split_ratio (list, optional): The ratio for splitting the image. Defaults to a list of common split ratios.
        Returns:
            BatchFeature: A BatchFeature object containing:
                - 'pixel_values': Tensor of processed image pixel values.
                - 'pixel_mask': Boolean pixel mask. This mask is a 2D tensor of shape (max_size, max_size) where:
                    - True (1) values indicate pixels that belong to the original resized image.
                    - False (0) values indicate pixels that are part of the padding.
                  The mask helps distinguish between actual image content and padded areas in subsequent processing steps.
                - 'num_crops': Tensor of the number of crops for each image.
        """
        max_size = self.max_image_size if max_image_size is None else max_image_size
        min_size = self.min_image_size if min_image_size is None else min_image_size

        if max_size not in [490, 980]:
            raise ValueError("max_image_size must be either 490 or 980")

        if isinstance(images, Image.Image):
            images = [images]

        pixel_values = []
        pixel_masks = []
        num_crops = []

        for image in images:
            crop_images = _split_image(image, split_image, split_ratio, max_size)
            num_crops.append(torch.tensor(len(crop_images)))
            for crop_image in crop_images:
                img_padded, pixel_mask = keep_ratio_resize_and_pixel_mask(
                    crop_image, max_size, min_size
                )
                img_padded = self.transform(img_padded)
                pixel_values.append(img_padded)
                pixel_masks.append(pixel_mask)

        return BatchFeature(
            data={
                "pixel_values": torch.stack(pixel_values),
                "pixel_mask": torch.stack(pixel_masks),
                "num_crops": torch.stack(num_crops),
            },
            tensor_type=return_tensors,
        )

    def preprocess(
        self,
        images,
        max_image_size=None,
        min_image_size=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        split_image: Optional[bool] = False,
        split_ratio: Optional[List[List[int]]] = [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 2],
            [2, 1],
            [3, 1],
            [4, 1],
        ],
    ):
        return self.__call__(
            images,
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            return_tensors=return_tensors,
            split_image=split_image,
            split_ratio=split_ratio,
        )
