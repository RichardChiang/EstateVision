from matplotlib import pyplot as plt

N_COLS = 5


def display_multiple_images(images, n_cols=N_COLS, fig_size=3):
    """
    Displays variable length of images in a grid with matplotlib.
    """
    n_rows = (len(images) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * fig_size, n_rows * fig_size),
    )
    fig.tight_layout()

    for i, image in enumerate(images):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(image)
        ax.set_axis_off()

    for j in range(i, n_rows * n_cols):
        ax = axes[j // n_cols, j % n_cols]
        ax.set_axis_off()

    plt.show()
