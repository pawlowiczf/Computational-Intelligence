import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# import plotly.io as pio
# pio.renderers.default = "browser"

from clonalg.problems.problem import Problem


def prepare_mesh_grid(
    problem: Problem,
    bounds: tuple[float, float] = (-5.5, 5.5),
    grid_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.linspace(bounds[0], bounds[1], grid_size)
    y_vals = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            xy = np.array([X[i, j], Y[i, j]])
            Z[i, j] = problem(xy)

    return X, Y, Z, x_vals, y_vals


def plot_3d_surface(
    problem: Problem,
    title=None,
    grid_size: int = 50,
):
    X, Y, Z, _, _ = prepare_mesh_grid(problem=problem, grid_size=grid_size)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    # if title is not None:
    #     ax.set_title(title)
    # else:
    #     ax.set_title(problem.__class__.__name__)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def plot_3d_surface_without_grid(
    problem: Problem,
    grid_size: int = 50,
):
    X, Y, Z, _, _ = prepare_mesh_grid(problem=problem, grid_size=grid_size)

    fig = go.Figure(
        data=go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="f(x, y)", thickness=15, len=0.6),
        )
    )

    axis_config = dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
        ticks="",
        title="",
        showbackground=False,
        visible=False,
    )

    fig.update_layout(
        title=problem.__class__.__name__,
        width=800,
        height=700,
        scene=dict(
            xaxis=axis_config,
            yaxis=axis_config,
            zaxis=axis_config,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),  # starting camera angle
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.show()


def plot_contour_and_paths(
    problem: Problem,
    paths: list[np.ndarray],
    grid_size: int = 200,
    title: str = "",
):
    """
    Create an interactive contour plot of a 2D function and overlay multiple optimization paths.

    Args:
        problem: An instance of a Problem class.
        paths: List of numpy arrays; each array is of shape (epochs, 2) containing an optimization trajectory.
        title: Title for the plot.
    """
    _, _, Z, x_vals, y_vals = prepare_mesh_grid(problem, grid_size=grid_size)

    fig = go.Figure(
        data=go.Contour(
            x=x_vals,
            y=y_vals,
            z=Z,
            colorscale="Viridis",
            contours=dict(showlines=False),
            colorbar=dict(title="Function Value"),
        )
    )

    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "pink",
        "brown",
    ]

    for idx, path in enumerate(paths):
        color_idx = idx % len(colors)
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(width=2, color=colors[color_idx]),
                # name=f"Run {idx+1}"
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=700,
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()
