import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 

__all__ = [
    "plot_paths",
    "plot_final_distribution",
    "plot_price_surface_3d",
    "plot_sensitivity_bars",
]


def _apply_dark_theme(ax: Axes) -> None:
    fig = ax.get_figure()
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    if hasattr(ax, "zaxis"):
        ax.tick_params(axis="z", colors="white")


def _maybe_save_and_show(
    fig: Figure,
    save_path: Optional[str | Path],
    show: bool,
) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()


def _colored_line(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    cmap: str = "hot",
    linewidth: float = 2.0,
) -> Optional[LineCollection]:
    """Add a 2D line whose color varies along y using a colormap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    points = np.column_stack([x, y]).reshape(-1, 1, 2)

    if len(points) < 2:
        return None

    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_max == y_min:
        y_min -= 0.5
        y_max += 0.5

    norm = Normalize(y_min, y_max)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y)
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)
    return lc


def _vertical_colored_line(
    ax: Axes,
    x_val: float,
    y_min: float,
    y_max: float,
    cmap: str = "hot",
    linewidth: float = 2.0,
    n_segments: int = 256,
) -> LineCollection:
    """Add a vertical line with a colormap gradient along y."""
    y = np.linspace(y_min, y_max, n_segments)
    x = np.full_like(y, float(x_val))

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(y_min, y_max)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y)
    lc.set_linewidth(linewidth)

    ax.add_collection(lc)
    return lc


def plot_paths(
    V: np.ndarray,
    r: Optional[np.ndarray] = None,
    max_paths: int = 100,
    show: bool = False,
    save_path: Optional[str | Path] = None,
    show_mean: bool = True,
    cmap: str = "hot",
) -> Figure:
    """
    Plot Monte Carlo paths for the property value.

    Parameters:
    V : ndarray, shape (M+1, I)
        Simulated property values.
    r : ndarray or None
        Short rate paths (unused, kept for API symmetry).
    max_paths : int
        Maximum number of individual paths to draw.
    show : bool
        If True, call plt.show().
    save_path : str | Path or None
        If not None, save the figure.
    show_mean : bool
        If True, overlay the mean path as a gradient-colored line.
    cmap : str
        Matplotlib colormap name for the highlighted mean path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(ax)

    num_paths = min(max_paths, V.shape[1])
    for i in range(num_paths):
        ax.plot(V[:, i], alpha=0.5, color="gray")

    if show_mean:
        mean_path = np.mean(V, axis=1)
        x = np.arange(V.shape[0])
        _colored_line(ax, x, mean_path, cmap=cmap, linewidth=2.0)

    ax.set_xlabel("Time step", color="white")
    ax.set_ylabel("Real Estate Value", color="white")

    _maybe_save_and_show(fig, save_path, show)
    return fig


def plot_final_distribution(
    V: np.ndarray,
    bins: int = 50,
    show: bool = False,
    save_path: Optional[str | Path] = None,
    show_mean: bool = True,
    cmap: str = "hot",
) -> Figure:
    """
    Plot histogram of terminal property values V_T.

    Parameters:
    V : ndarray, shape (M+1, I)
        Simulated property values.
    bins : int
        Number of histogram bins.
    show : bool
        If True, call plt.show().
    save_path : str | Path or None
        If not None, save the figure.
    show_mean : bool
        If True, overlay a vertical gradient line at the mean terminal value.
    cmap : str
        Matplotlib colormap name for the highlighted mean line.
    """
    V_T = V[-1, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(ax)

    ax.hist(V_T, bins=bins, alpha=0.75, color="gray")

    if show_mean:
        mean_VT = float(np.mean(V_T))
        ymin, ymax = ax.get_ylim()

        _vertical_colored_line(ax, mean_VT, ymin, ymax, cmap=cmap, linewidth=2.0)
    
    ax.set_xlabel("Final Real Estate Value", color="white")
    ax.set_ylabel("Frequency", color="white")

    _maybe_save_and_show(fig, save_path, show)
    return fig


def plot_price_surface_3d(
    K_mesh: np.ndarray,
    T_mesh: np.ndarray,
    P: np.ndarray,
    show: bool = False,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    3D surface of option price P(K, T).
    Parameters:
    K_mesh, T_mesh : 2D arrays
        Meshgrids of strike and maturity.
    P : 2D array
        Prices corresponding to each (K, T) pair.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    _apply_dark_theme(ax)
    surf = ax.plot_surface(
        K_mesh,
        T_mesh,
        P,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        cmap="hot",
        alpha=0.95,
    )
    ax.set_xlabel("Strike K", color="white")
    ax.set_ylabel("Maturity T (years)", color="white")
    ax.set_zlabel("Option price", color="white")
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Price', color='white')
    cbar.ax.tick_params(labelcolor='white')
    ax.view_init(elev=25, azim=-135)
    _maybe_save_and_show(fig, save_path, show)
    return fig

def plot_sensitivity_bars(
    param_names: list[str],
    price_low: list[float] | np.ndarray,
    price_high: list[float] | np.ndarray,
    base_price: float,
    show: bool = False,
    save_path: Optional[str | Path] = None,
    highlight_base: bool = True,
    cmap: str = "hot",
) -> Figure:
    """
    Horizontal bar / tornado-style chart of one-way parameter sensitivities.

    Parameters:
    param_names : list[str]
        Parameter names corresponding to each bar.
    price_low : array-like
        Option prices with the parameter bumped down.
    price_high : array-like
        Option prices with the parameter bumped up.
    base_price : float
        Option price at the current/base parameter set.
    show : bool
        If True, call plt.show().
    save_path : str | Path or None
        If not None, save the figure.
    highlight_base : bool
        If True, draw a vertical gradient line at the base price.
    cmap : str
        Matplotlib colormap name for the highlighted base line.
    """
    low_arr = np.asarray(price_low, dtype=float)
    high_arr = np.asarray(price_high, dtype=float)

    if low_arr.shape != high_arr.shape:
        raise ValueError("price_low and price_high must have the same shape.")
    if low_arr.ndim != 1:
        raise ValueError("price_low and price_high must be 1D arrays.")
    if len(param_names) != low_arr.shape[0]:
        raise ValueError("param_names and price arrays must have the same length.")

    delta = np.abs(high_arr - low_arr)
    order = np.argsort(delta)[::-1]

    names_sorted = [param_names[i] for i in order]
    low_sorted = low_arr[order]
    high_sorted = high_arr[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    _apply_dark_theme(ax)

    y_pos = np.arange(len(names_sorted))

    for i, (lo, hi) in enumerate(zip(low_sorted, high_sorted)):
        left = min(lo, hi)
        width = abs(hi - lo)
        ax.barh(y_pos[i], width, left=left, alpha=0.8, color="gray")

    if highlight_base:
        ymin, ymax = ax.get_ylim()

        _vertical_colored_line(ax, base_price, ymin, ymax, cmap=cmap, linewidth=1.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, color="white")
    ax.set_xlabel("Option price", color="white")

    _maybe_save_and_show(fig, save_path, show)
    return fig