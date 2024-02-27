from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "grid"])


def sized_subplots(
    n_axes: int = 1, n_cols: int = 1, ax_size: tuple[int, int] = (5, 4)
) -> tuple[plt.Figure, np.ndarray]:
    n_rows = ceil(n_axes / n_cols)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(ax_size[0] * n_cols, ax_size[1] * n_rows),
        squeeze=False,
    )
    return fig, axs
