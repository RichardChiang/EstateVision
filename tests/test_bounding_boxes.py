from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from src.bounding_boxes import (
    _filter_regions,
    _get_roof_color,
    _replace_roof_colors,
    find_roof_boxes,
)


@dataclass
class Region:
    area: int
    bbox: Tuple[int, int, int, int]


def test_filter_regions_default_min_area():
    regions = [
        Region(100, (0, 0, 0, 0)),
        Region(200, (0, 0, 0, 0)),
        Region(300, (0, 0, 0, 0)),
    ]
    assert [region.area for region in _filter_regions(regions)] == [200, 300]


def test_get_roof_color_gets_second_most_common_color():
    gray_image = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0.98, 0.98, 0.98, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert _get_roof_color(gray_image) == 0.98


def test_get_roof_color_with_empty_image_raises():
    gray_image = np.array([])
    with pytest.raises(BaseException):
        _get_roof_color(gray_image)


def test_replace_roof_colors():
    H = 0.98  # house color
    gray_image = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, H, H, H, 0],
            [0, H, H, H, 0],
            [0, H, H, H, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(
        _replace_roof_colors(gray_image, H),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]
        ),
    )


def test_replace_roof_colors_with_similar_colors():
    gray_image = np.array([[0.1, 0.919, 0.92, 0.921, 0.1]])
    assert np.array_equal(
        _replace_roof_colors(gray_image, 0.92, epsilon=0.01),
        np.array([[1, 0, 0, 0, 1]]),
    )


def test_find_roof_boxes():
    image, bboxes = find_roof_boxes("data/street_map.png")
    assert len(image.shape) == 3
    assert len(bboxes) == 51
