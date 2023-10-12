import numpy as np
import pytest

from src.postprocessing import crop_image, save_images


class FakeFilesystem:
    def __init__(self):
        self.files = {}

    def imsave(self, image_path, image):
        self.files[image_path] = image


@pytest.fixture
def filesystem():
    return FakeFilesystem()


def test_crop_image():
    image = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(
        crop_image(image, 1, 1, 3, 3, buffer=0),
        np.array(
            [
                [0.98, 0.98, 0.98],
                [0.98, 0.98, 0.98],
                [0.98, 0.98, 0.98],
            ]
        ),
    )


def test_crop_image_including_buffer():
    image = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(
        crop_image(image, 1, 1, 3, 3, buffer=1),
        image,
    )


def test_crop_image_with_buffer_doesnt_exceed_edge():
    image = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(
        crop_image(image, 1, 1, 3, 3, buffer=100),
        image,
    )


def test_save_image(filesystem):
    image = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    save_images(filesystem, [image], "dir", "original_image")
    assert "dir/original_image_0.jpg" in filesystem.files


def test_save_multiple_images(filesystem):
    image = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    save_images(filesystem, [image, image], "dir", "original_image")
    assert "dir/original_image_0.jpg" in filesystem.files
    assert "dir/original_image_1.jpg" in filesystem.files
