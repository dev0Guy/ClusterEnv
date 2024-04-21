from clusterenv._types import JobStatus
from typing import Iterable, Callable, Any
from dataclasses import dataclass, field
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
import math

from numba import njit, jit

CMAP_COLORS: tuple = ("copper", "gray", "twilight", "summer")
REGULAR_COLOR: str = "copper"
REGULAR_TITLE_COLOR: str = "black"
ERROR_COLOR: str = "RdGy"
CORRECT_COLOR: str = "Greens"
ERROR_TITLE_COLOR: str = "red"


def _hide_unused(axs: Axes,  nodes: np.array, jobs: np.array, nodes_n_columns: int):
    nodes_to_remove: list = axs[:, :nodes_n_columns].flatten()[nodes:]
    jobs_to_remove: list = axs[:, nodes_n_columns:].flatten()[jobs:]
    for ax in nodes_to_remove:
        plt.delaxes(ax)
    for ax in jobs_to_remove:
        plt.delaxes(ax)


@jit(forceobj=True)
def _draw(
    matrix: np.ndarray,
    /,
    *,
    title: str,
    title_color: str,
    ax: Axes,
    time: int,
    resource: int,
    cmap: str,
):
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100)
    ax.set_title(title, fontsize=10, color=title_color)
    ax.set_xticks(np.arange(0, time, 0.5), minor=True)
    ax.set_yticks(np.arange(0, resource, 0.5), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])


@jit(forceobj=True)
def _draw_job(
    job: np.ndarray,
    /,
    *,
    title_color: str,
    idx: int,
    ax: Axes,
    time: int,
    resource: int,
    cmap: str,
    status: JobStatus,
):
    title: str = f"[J.{idx}]"
    cmap = cmap if cmap == ERROR_COLOR else CMAP_COLORS[status]
    _draw(
        job,
        title=title,
        title_color=title_color,
        ax=ax,
        time=time,
        resource=resource,
        cmap=cmap,
    )


@jit(forceobj=True)
def _draw_node(
    node: np.ndarray,
    /,
    *,
    title_color: str,
    idx: int,
    ax: Axes,
    time: int,
    resource: int,
    cmap: str,
):
    title: str = f"[N.{idx}]"
    _draw(
        node,
        title=title,
        title_color=title_color,
        ax=ax,
        time=time,
        resource=resource,
        cmap=cmap,
    )


def color_map(error, correct):
    def inner(idx, pos):
        if error and idx == error[pos]:
            return ERROR_COLOR
        elif correct and idx == correct[pos]:
            return CORRECT_COLOR
        return REGULAR_COLOR
    return inner


@dataclass(slots=False)
class ClusterRenderer:
    """
    Craete Gym Cluster Object.
    Allow To Represent Cluster logic: Scheduling, TimeTick, Generating Job.
    Job & Node reperesentation should be similar.

    :param nodes: List of nodes repersentation
    :param jobs: List of job representation
    :param resource: Number of Job/Node resource type
    :param max_time: Max Job Run time
    :param cooldown: Render cooldown between stepes
    """

    nodes: int
    jobs: int
    resource: int
    time: int
    cooldown: float
    fig: Figure = field(init=False)
    axs: npt.NDArray = field(init=False)

    def __post_init__(self):
        self.jobs_n_columns: int = math.ceil(self.jobs**0.5)
        self.nodes_n_columns: int = math.ceil(self.nodes**0.5)

        jobs_n_rows: int = math.ceil(self.jobs / self.jobs_n_columns)
        nodes_n_rows: int = math.ceil(self.nodes / self.nodes_n_columns)

        n_rows: int = max(jobs_n_rows, nodes_n_rows)
        n_columns: int = self.nodes_n_columns + self.jobs_n_columns

        self.fig, self.axs = plt.subplots(
            n_rows, n_columns, figsize=(8, 4), facecolor="white"
        )
        _hide_unused(
            self.axs,
            nodes=self.nodes,
            jobs=self.jobs,
            nodes_n_columns=self.nodes_n_columns,
        )

    def __call__(
        self,
        obs: dict[str, np.ndarray],
        /,
        *,
        current_time: int,
        error: None | tuple[int, int],
        correct: None | tuple[int, int]
    ) -> Any:
        self.fig.suptitle(f"Time: {current_time}", fontsize=16, fontweight="bold")
        nodes: npt.NDArray = obs["Usage"]
        queue: npt.NDArray = obs["Queue"]
        status: npt.NDArray[np.uint32] = obs["Status"]
        cmap_color = color_map(error, correct)
        title_color: Callable[[int, int], str] = (
            lambda idx, pos: ERROR_TITLE_COLOR
            if error and idx == error[pos]
            else REGULAR_TITLE_COLOR
        )
        node_ax: Callable[[int], npt.NDArray] = lambda n_idx: self.axs[
            n_idx // self.nodes_n_columns, n_idx % self.nodes_n_columns
        ]
        job_ax: Callable[[int], npt.NDArray] = lambda j_idx: self.axs[
            j_idx // self.jobs_n_columns,
            self.nodes_n_columns + (j_idx % self.jobs_n_columns),
        ]
        # update matries
        for n_idx, node in enumerate(nodes):
            _draw_node(
                node,
                title_color=title_color(n_idx, 0),
                idx=n_idx,
                ax=node_ax(n_idx),
                time=self.time,
                resource=self.resource,
                cmap=cmap_color(n_idx, 0),
            )
        for j_idx, job in enumerate(queue):
            _draw_job(
                job,
                title_color=title_color(j_idx, 1),
                idx=j_idx,
                ax=job_ax(j_idx),
                time=self.time,
                resource=self.resource,
                cmap=cmap_color(j_idx, 1),
                status=JobStatus(status[j_idx]),
            )
        # update figure
        plt.draw()
        plt.pause(self.cooldown)
        return self.fig