import numpy as np

from src.crop_images import crop_image


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
