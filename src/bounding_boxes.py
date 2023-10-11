from collections import Counter

from matplotlib import pyplot as plt
from skimage import feature, filters, io, measure
from skimage.color import rgb2gray


def _filter_regions(regions, min_area=200):
    """
    Filters out regions that are too small. Minimum area defaults to 200 to avoid
    noise from capturing sheds, small roofs and road objects.
    """
    return [region for region in regions if region.area >= min_area]


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


def load_image(image_path):
    """
    Loads image with skimage.
    """
    return io.imread(image_path)


def find_roof_boxes(image_path):
    """
    Finds the bounding boxes around the houses in the image.
    """
    image = load_image(image_path)

    gray_image = rgb2gray(image)
    gray_image = _replace_roof_colors(
        gray_image,
        _get_roof_color(gray_image),
    )

    # smooth edges
    edges = filters.gaussian(gray_image, sigma=1)

    # Apply Canny edge detection
    edges = feature.canny(edges, sigma=1)

    # Join overlapping edges
    regions = measure.regionprops(measure.label(edges))

    # Filter out regions that are too small
    filtered_regions = _filter_regions(regions)

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
