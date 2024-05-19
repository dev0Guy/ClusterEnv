from dataclasses import dataclass, field
from typing import Self
from enum import IntEnum
import math

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from gymnasium.core import ActType, ObsType, RenderFrame
from pydantic import PositiveInt
from typing import TypeVar, Generic, SupportsFloat, Any
import gymnasium as gym
import numpy as np

ShapType = TypeVar("ShapType", bound=tuple)
DType = TypeVar("DType", bound=np.dtype)

NodeIndex = TypeVar("NodeIndex", bound=PositiveInt)
JobIndex = TypeVar("JobIndex", bound=PositiveInt)


class JobStatusScheduleError(ValueError):
    def __init__(self, msg, *args, **kwargs):
        super(JobStatusScheduleError, self).__init__(msg, *args, **kwargs)


class NotEnoughResourceError(ValueError):
    def __init__(self, msg, *args, **kwargs):
        super(NotEnoughResourceError, self).__init__(msg, *args, **kwargs)


class Array(np.ndarray, Generic[DType, ShapType]):
    pass


class Status(IntEnum):
    NONEXISTENT = 0
    PENDING = 1
    RUNNING = 2
    COMPLETE = 3


@dataclass
class Nodes:
    spec: Array[np.float64, 3]
    usage: Array[np.float64, 3] | None = field(init=False)

    def __post_init__(self) -> None:
        self.usage = np.zeros_like(self.spec)

    def tick(self, time: PositiveInt) -> None:
        self.usage = np.roll(self.usage, -1, axis=-1)
        self.usage[:, :, -1] = 0

    def __len__(self) -> PositiveInt:
        return self.spec.shape[0]


@dataclass
class Jobs:
    spec: Array[np.float64, 3]
    submission: Array[np.uint]
    status: Array[Status] = field(init=False)
    length: Array[np.uint] = field(init=False)
    run_time: Array[np.uint] = field(init=False)
    wait_time: Array[np.uint] = field(init=False)

    def __post_init__(self) -> None:
        self.status = np.full(self.spec.shape[0], Status.PENDING, dtype=np.uint)
        length_by_resource: Array[np.bool_, 2] = self.spec.shape[-1] - (
            self.spec[:, :, ::-1] != 0
        ).argmax(axis=-1)
        self.length = length_by_resource.max(axis=-1)
        self.run_time = np.zeros(self.spec.shape[0])
        self.wait_time = np.zeros(self.spec.shape[0])

    def tick(self, time: PositiveInt) -> None:
        nonexistent: Array[bool] = self.status == Status.NONEXISTENT
        pending: Array[bool] = self.status == Status.PENDING
        running: Array[bool] = self.status == Status.RUNNING
        change_to_complete: Array[bool] = self.length == self.run_time
        change_to_pending: Array[bool] = np.logical_and(
            self.submission == time, nonexistent
        )
        self.status[change_to_pending] = Status.PENDING
        self.status[change_to_complete] = Status.COMPLETE
        self.wait_time[pending] += 1
        self.run_time[running] += 1

    def __len__(self) -> PositiveInt:
        return self.spec.shape[0]

    @staticmethod
    def generate(
        n: PositiveInt, r: PositiveInt, t: PositiveInt, *, max_usage: PositiveInt
    ) -> Self:
        ARRIVAL = dict(option=[0.0], prob=[1])
        USAGE = dict(option=[1], prob=[1])
        LENGTH = dict(option=[0, 0.5, 0.7], prob=[0.1, 0.5, 0.4])
        submission: Array[np.uint] = t * np.random.choice(
            ARRIVAL["option"], size=n, p=ARRIVAL["prob"]
        ).astype(np.uint)
        length: Array[np.uint] = (
            1 + t * np.random.choice(LENGTH["option"], size=(n, r), p=LENGTH["prob"])
        ).astype(np.uint)
        spec: np.array = max_usage * np.random.choice(
            USAGE["option"], size=n, p=USAGE["prob"]
        ).astype(np.float32)
        spec = np.tile(spec[..., np.newaxis, np.newaxis], (r, t))
        mask = np.arange(t) >= length[..., np.newaxis]
        spec[mask] = 0.0
        return Jobs(spec=spec, submission=submission)


class ClusterEnv(gym.Env):
    metadata: dict = {"render_modes": ["human", "rgb_array", ""], "render_fps": 4}

    def __init__(
        self,
        nodes: PositiveInt,
        jobs: PositiveInt,
        resource: PositiveInt,
        time: PositiveInt,
        *,
        max_episode_steps: PositiveInt,
        render_mode: str = "",
    ):
        super(ClusterEnv, self).__init__()
        nodes_shape: tuple[int, int, int] = (nodes, resource, time)
        jobs_shape: tuple[int, int, int] = (jobs, resource, time)
        self.nodes = Nodes(np.ones(nodes_shape, dtype=np.float64))  # TODO: create nodes
        self.jobs = Jobs.generate(jobs, resource, time, max_usage=1)
        self.current_time: PositiveInt = 0
        self.last_action: PositiveInt = 0
        # gym config
        self.render_mode: str = render_mode
        self.max_episode_steps: int = max_episode_steps
        self.n_steps: PositiveInt = 0
        self.observation_space = gym.spaces.Dict(
            dict(
                Nodes=gym.spaces.Box(low=0, high=np.inf, shape=self.nodes.spec.shape),
                Jobs=gym.spaces.Box(low=0, high=np.inf, shape=self.jobs.spec.shape),
            )
        )
        self.action_space = gym.spaces.Discrete(nodes * jobs + 1)
        # render config
        if self.render_mode:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError(f"Rendering env requires matplotlib to be installed.")

    def _convert_action(self, action: PositiveInt) -> tuple[NodeIndex, JobIndex]:
        n_nodes: PositiveInt = len(self.nodes)
        return action % n_nodes, action // n_nodes

    def _schedule(self, n_idx: NodeIndex, j_idx: JobIndex) -> None:
        if self.jobs.status[j_idx] != Status.PENDING:
            raise JobStatusScheduleError(
                f"Schedule operation with incorrect status: {Status(self.jobs.status[j_idx]).name}. Only PENDING are allowed"
            )
        node_can_contain_job: bool = np.all(
            (self.nodes.spec[n_idx] - self.nodes.usage[n_idx]) >= self.jobs.spec[j_idx]
        )
        if not node_can_contain_job:
            raise NotEnoughResourceError(f"Schedule operation with incorrect usage")
        self.nodes.usage[n_idx] += self.jobs.spec[j_idx]
        self.jobs.status[j_idx] = Status.RUNNING

    def _sort_jobs(self) -> Array[np.float64, 3]:
        x = self.jobs.status
        x[self.jobs.status == Status.NONEXISTENT] = Status.COMPLETE + 1
        x = np.argsort(self.jobs.status)
        return self.jobs.spec[x]

    def _get_observation(self) -> dict:
        jobs = self.jobs.spec
        jobs[self.jobs.status == Status.RUNNING] = 0
        jobs[self.jobs.status == Status.COMPLETE] = 0
        jobs[self.jobs.status == Status.NONEXISTENT] = 0
        free_space = self.nodes.spec - self.nodes.usage
        return dict(
            Nodes=free_space.astype(np.float32),
            Jobs=self._sort_jobs().astype(np.float32),
        )

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.last_action = action
        self.n_steps += 1
        reward = -1
        if action == 0:
            self.jobs.tick(self.current_time)
            self.nodes.tick(self.current_time)
            self.current_time += 1
        else:
            try:
                self._schedule(*self._convert_action(action - 1))
            except JobStatusScheduleError as status_error:
                self.last_action *= -1
                reward = -10
            except NotEnoughResourceError as not_enough_resource:
                self.last_action *= -1
                reward = -10
        n_jobs: PositiveInt = len(self.jobs)
        running: PositiveInt = np.count_nonzero(self.jobs.status == Status.RUNNING)
        completed: PositiveInt = np.count_nonzero(self.jobs.status == Status.COMPLETE)
        trounced: bool = (
            n_jobs == (completed + running) or self.n_steps >= self.max_episode_steps
        )
        self.render()
        return self._get_observation(), reward, n_jobs == completed, trounced, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.last_action = 0
        self.n_steps = 0
        self.current_time = 0
        self.nodes.usage[:] = 0
        self.jobs.status[:] = Status.NONEXISTENT
        self.jobs.run_time[:] = 0
        self.jobs.wait_time[:] = 0
        self.jobs.tick(0)
        if self.render_mode:
            n_jobs: PositiveInt = len(self.jobs)
            n_nodes: PositiveInt = len(self.nodes)

            self.jobs_n_columns: int = math.ceil(n_jobs**0.5)
            self.nodes_n_columns: int = math.ceil(n_nodes**0.5)

            jobs_n_rows: int = math.ceil(n_jobs / self.jobs_n_columns)
            nodes_n_rows: int = math.ceil(n_nodes / self.nodes_n_columns)

            n_rows: int = max(jobs_n_rows, nodes_n_rows)
            n_columns: int = self.nodes_n_columns + self.jobs_n_columns
            self._fig, self._axs = plt.subplots(
                n_rows, n_columns, figsize=(8, 4), facecolor="white"
            )
            self._hide_unused_subplot()
        self.render()
        return self._get_observation(), {}

    def _hide_unused_subplot(self):
        nodes_to_remove: list = self._axs[:, : self.nodes_n_columns].flatten()[
            len(self.nodes) :
        ]
        jobs_to_remove: list = self._axs[:, self.nodes_n_columns :].flatten()[
            len(self.jobs) :
        ]
        for ax in nodes_to_remove:
            plt.delaxes(ax)
        for ax in jobs_to_remove:
            plt.delaxes(ax)

    def _get_node_cell(self, index: PositiveInt) -> plt.Axes:
        return self._axs[index // self.nodes_n_columns, index % self.nodes_n_columns]

    def _get_job_cell(self, index: PositiveInt) -> plt.Axes:
        return self._axs[
            index // self.jobs_n_columns,
            self.nodes_n_columns + (index % self.jobs_n_columns),
        ]

    def _draw_cell(
        self, cell: Array[np.float64, 2], title: str, ax: plt.Axes, *, cmap: str
    ) -> None:
        ax.imshow(cell, vmin=0, vmax=255, cmap=cmap)
        ax.set_title(title, fontsize=10, color="black")
        ax.set_xticks(np.arange(0.5, cell.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(0.5, cell.shape[0], 1), minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid(which="both", color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if not self.render_mode:
            return

        observation: dict = self._get_observation()
        self._fig.suptitle(f"Time: {self.current_time}", fontsize=16, fontweight="bold")
        nodes: Array[np.float64, 3] = (1 - observation["Nodes"]) * 255
        jobs: Array[np.float64, 3] = observation["Jobs"] * 255
        not_zero: bool = self.last_action != 0
        action_n_idx = action_j_idx = -1
        action_color = "copper"
        if not_zero:
            action_n_idx, action_j_idx = self._convert_action(abs(self.last_action - 1))
            action_color = "Greens" if self.last_action > 0 else "RdGy"

        for n_idx, node in enumerate(nodes):
            color: str = "copper" if n_idx != action_n_idx else action_color
            self._draw_cell(
                node, f"Node.{n_idx}", self._get_node_cell(n_idx), cmap=color
            )

        for j_idx, job in enumerate(jobs):
            color: str = "copper" if j_idx != action_j_idx else action_color
            self._draw_cell(job, f"Job.{j_idx}", self._get_job_cell(j_idx), cmap=color)

        plt.draw()
        plt.pause(1e-7)
        if self.render_mode == "rgb_array":
            buf = self._fig.canvas.tostring_rgb()
            width, height = self._fig.canvas.get_width_height()
            expected_height: int = int(self._fig.get_figheight() * self._fig.dpi)
            expected_width: int = int(self._fig.get_figwidth() * self._fig.dpi)
            width_mult: int = expected_width // width
            height_mult: int = expected_height // height
            return np.frombuffer(buf, dtype=np.uint8).reshape(
                height * height_mult, width * width_mult, 3
            )

    def close(self):
        plt.close(self._fig)
