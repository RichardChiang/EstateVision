from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from src.bounding_boxes import (
    _filter_border_regions,
    _filter_google_maps_logo,
    _filter_small_regions,
    _get_roof_color,
    _replace_roof_colors,
    find_roof_boxes,
)
from src.utils import load_image


@dataclass
class Region:
    area_bbox: int
    bbox: Tuple[int, int, int, int]


@dataclass
class Image:
    shape: Tuple[int, int]


def test_filter_regions_default_min_area():
    regions = [
        Region(100, (0, 0, 0, 0)),
        Region(200, (0, 0, 0, 0)),
        Region(300, (0, 0, 0, 0)),
    ]
    assert [region.area_bbox for region in _filter_small_regions(regions, 200)] == [
        200,
        300,
    ]


def test_filter_border_regions_removes_border_bboxes():
    regions = [
        Region(100, (0, 0, 0, 0)),  # overlaps
        Region(200, (50, 50, 90, 100)),  # overlaps
        Region(300, (50, 50, 90, 90)),
    ]
    image = Image((100, 100))
    filtered = _filter_border_regions(regions, image)

    assert len(filtered) == 1
    assert filtered[0].bbox == (50, 50, 90, 90)


def test_filter_border_regions_with_bbox_overlapping_with_buffer():
    regions = [
        Region(100, (0, 0, 0, 0)),  # overlaps
        Region(200, (50, 50, 90, 100)),  # overlaps
        Region(300, (50, 50, 90, 90)),  # overlaps
    ]
    image = Image((100, 100))
    filtered = _filter_border_regions(regions, image, buffer=10)
    assert len(filtered) == 0


def test_filter_bbox_overlapping_with_google_maps_logo():
    regions = [
        Region(100, (0, 0, 0, 0)),
        Region(200, (0, 0, 890, 890)),
        Region(300, (50, 50, 990, 990)),  # overlaps
    ]
    image = Image((1000, 1000))
    filtered = _filter_google_maps_logo(regions, image)
    assert [region.bbox[-1] for region in filtered] == [0, 890]


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
    image = load_image("data/street_map.png")
    image, bboxes = find_roof_boxes(image)
    assert len(image.shape) == 3
    assert len(bboxes) >= 51
