from collections import Counter

from matplotlib import pyplot as plt
from skimage import feature, filters, measure
from skimage.color import rgb2gray


def _filter_small_regions(regions, min_area=500):
    """
    Removes regions that are too small or have unrealistic dimensions. Minimum area
    defaults to avoid noise from capturing sheds, small roofs and road objects.
    """
    return [region for region in regions if region.area_bbox >= min_area]


def _filter_border_regions(regions, image, buffer=5):
    """
    Removes regions that are too close to the edge of the image. Removes regions cut
    off by the edge of the image.
    """
    return [
        region
        for region in regions
        if region.bbox[0] > buffer
        and region.bbox[1] > buffer
        and region.bbox[2] < image.shape[0] - buffer
        and region.bbox[3] < image.shape[1] - buffer
    ]


def _filter_google_maps_logo(regions, image):
    """
    Removes regions that overlap with the bottom of the image because copyright logo is
    often placed there.
    """
    return [region for region in regions if region.bbox[2] < image.shape[1] - 100]


def _get_roof_color(gray_image):
    """
    Gets the color identifying roofs in the gray image. Most common color is the white
    of the background, followed by the gray of the houses.
    """
    top_colors = Counter(gray_image.flat).most_common(3)
    if not top_colors:
        raise BaseException("No colors found in image")
    return top_colors[1][0]


def _replace_roof_colors(image, house_color, epsilon=0.0001):
    """
    Replaces background colors with black and house colors with white. Colors within
    epsilon of the house color are considered the same color.
    """
    BLACK, WHITE = 1, 0
    image[abs(image - house_color) > epsilon] = BLACK
    image[abs(image - house_color) <= epsilon] = WHITE
    return image


def find_roof_boxes(image):
    """
    Finds the bounding boxes around the houses in the image.
    """
    gray_image = rgb2gray(image)
    gray_image = _replace_roof_colors(
        gray_image,
        _get_roof_color(gray_image),
    )

    # smooth edges
    edges = filters.gaussian(gray_image, sigma=10)

    # Apply Canny edge detection
    edges = feature.canny(edges, sigma=1)

    # Join overlapping edges
    regions = measure.regionprops(measure.label(edges))

    # Filter out invalid regions
    filtered_regions = _filter_small_regions(regions)
    filtered_regions = _filter_border_regions(filtered_regions, image)
    filtered_regions = _filter_google_maps_logo(filtered_regions, image)

    # Get bounding boxes
    bboxes = [region.bbox for region in filtered_regions]

    return image, bboxes


def display_bounding_boxes(image, boxes):
    """
    Displays the bounding boxes around the houses in the image.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    ax.imshow(image)

    for box in boxes:
        min_row, min_col, max_row, max_col = box
        rect = plt.Rectangle(
            (min_col, min_row),
            max_col - min_col,
            max_row - min_row,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
        ax.add_patch(rect)

    plt.title("Bounding Boxes around Houses")
    plt.show()
