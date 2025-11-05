import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure


__all__ = ["plot_paths","plot_final_distribution","plot_rate_value_heatmap",]


def _apply_dark_theme(ax: Axes) -> None:
    fig = ax.get_figure()
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")


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


def plot_paths(
    V: np.ndarray,
    r: Optional[np.ndarray] = None,
    max_paths: int = 100,
    show: bool = False,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    plot monte carlo paths for the property value

    parameters:
    V : np.ndarray
        simulated property values with shape (M+1, I).
    r : np.ndarray, optional
        simulated short rate paths (currently unused).
    max_paths : int
        maximum number of paths to draw.
    show : bool
        if True, call plt.show() on the figure.
    save_path : str or Path, optional
        if provided, save the figure to this location.

    returns:
    figure, created by matplotlib
    
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(ax)

    num_paths = min(max_paths, V.shape[1])
    for i in range(num_paths):
        ax.plot(V[:, i], alpha=0.5, color="gray")

    ax.set_title("Simulated Real Estate Value Paths", color="white")
    ax.set_xlabel("Time step", color="white")
    ax.set_ylabel("Real Estate Value", color="white")

    _maybe_save_and_show(fig, save_path, show)
    return fig


def plot_final_distribution(
    V: np.ndarray,
    bins: int = 50,
    show: bool = False,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    plot histogram of final property values

    parameters:
    V : np.ndarray
        simulated property values with shape (M+1, I).
    bins : int
        number of histogram bins.
    show : bool
        if True, call plt.show() on the figure.
    save_path : str or Path, optional
        if provided, save the figure to this location.

    returns:
    figure, created by matplotlib

    """
    V_T = V[-1, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(ax)

    ax.hist(V_T, bins=bins, alpha=0.75, color="gray")
    ax.set_title("Distribution of Final Real Estate Values", color="white")
    ax.set_xlabel("Final Real Estate Value", color="white")
    ax.set_ylabel("Frequency", color="white")

    _maybe_save_and_show(fig, save_path, show)
    return fig


def plot_rate_value_heatmap(
    r: np.ndarray,
    V: np.ndarray,
    bins: int = 50,
    show: bool = False,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    plot a heatmap of the joint distribution of interest rates & values

    parameters:
    r : np.ndarray
        simulated short rate paths with shape (M+1, I).
    V : np.ndarray
        simulated property values with shape (M+1, I).
    bins : int
        number of bins along each axis.
    show : bool
        if True, call plt.show() on the figure.
    save_path : str or Path, optional
        if provided, save the figure to this location.

    returns:
    figure, created by matplotlib

    """
    r_flat = r.flatten()
    V_flat = V.flatten()
    heatmap, xedges, yedges = np.histogram2d(r_flat, V_flat, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(ax)

    im = ax.imshow(
        heatmap.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="gray",
    )
    ax.set_title("Heatmap of Interest Rates and Real Estate Values", color="white")
    ax.set_xlabel("Interest Rate", color="white")
    ax.set_ylabel("Real Estate Value", color="white")

    cbar = fig.colorbar(im, ax=ax, label="Frequency")
    cbar.ax.yaxis.set_tick_params(color="white")
    for label in cbar.ax.get_yticklabels():
        label.set_color("white")

    _maybe_save_and_show(fig, save_path, show)
    return fig


if __name__ == "__main__":
    from bsre_options import BSREOptions

    model = BSREOptions(
        V0=200000.0,
        K=220000.0,
        T=1.0,
        r0=0.03,
        kappa_r=0.3,
        theta_r=0.03,
        sigma_r=0.01,
        v0=0.02,
        kappa_v=1.0,
        theta_v=0.02,
        sigma_v=0.1,
        rho=-0.5,
        q=0.02,
        carry_cost=0.0,
        M=100,
        I=10000,
        seed=123,
    )
    V, r = model.simulate_paths()

    plot_paths(V, r, max_paths=100, show=True, save_path=Path("assets/paths.png"))
    plot_final_distribution(V, show=True, save_path=Path("assets/final_dist.png"))
    plot_rate_value_heatmap(r, V, show=True, save_path=Path("assets/rate_value_heatmap.png"))
