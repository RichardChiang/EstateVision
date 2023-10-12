from matplotlib import pyplot as plt
from skimage import io

N_COLS = 5


def load_image(image_path: str):
    """
    Loads image with skimage.
    """
    return io.imread(image_path)


def display_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def display_multiple_images(images, labels=None, n_cols=N_COLS, fig_size=3):
    """
    Displays variable length of images in a grid with matplotlib.
    """
    if not labels:
        labels = [None] * len(images)

    if len(images) == 1:
        display_image(images[0])

    n_cols = min(n_cols, len(images))
    n_rows = (len(images) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * fig_size, n_rows * fig_size),
    )
    fig.tight_layout()

    for i, image in enumerate(images):
        ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i]
        ax.set_title(labels[i])
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()

    for j in range(i, n_rows * n_cols):
        ax = axes[j // n_cols, j % n_cols] if n_rows > 1 else axes[j]
        ax.set_axis_off()

    plt.show()
