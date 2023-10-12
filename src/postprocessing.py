from typing import List

import numpy as np


def crop_image(image: np.ndarray, min_y, min_x, max_y, max_x, buffer=5):
    """
    Crop the image using NumPy array slicing. A buffer is added to each side of the
    cropped image to account for edge effects.
    """
    min_y = max(min_y - buffer, 0)
    min_x = max(min_x - buffer, 0)
    max_y = min(max_y + buffer, image.shape[0])
    max_x = min(max_x + buffer, image.shape[1])

    cropped_image = image[min_y : max_y + 1, min_x : max_x + 1]

    return cropped_image


def save_images(
    filesystem, images: List[np.ndarray], directory: str, parent_filename: str
):
    """
    Save images to filesystem.
    """
    for i, image in enumerate(images):
        image_path = f"{directory}/{parent_filename}_{i}.jpg"
        filesystem.imsave(image_path, image)
