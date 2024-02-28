from typing import Iterable, ParamSpecArgs, Self, Any, SupportsFloat, Optional
from dataclasses import dataclass, field
from gymnasium.core import RenderFrame
from typing_extensions import Callable, NamedTuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from .base import ClusterObject, JobStatus, Jobs
from numpy._typing import NDArray
import matplotlib.pyplot as plt
import numpy.typing as npt
import gymnasium as gym
import numpy as np
import logging
import math

@dataclass
class DistribConfig:
    options: list[Any]
    probability: list[float]

DEFUALT_ARRIVAL_FUNC: Callable = lambda: DistribConfig(options=[.0,.2,.3], probability=[.5,.4,.1])
DEFUALT_LENGTH_FUNC: Callable = lambda: DistribConfig(options=[.0,.2,.3], probability=[.7,.2,.1])
DEFUALT_USAGE_FUNC: Callable = lambda: DistribConfig(options=[0.1,0.5,1], probability=[.7,.2,.1])

@dataclass
class ClusterRenderer:
    nodes: int
    jobs: int
    resource: int
    time: int
    cooldown: float = field(default=5)
    fig: Figure = field(init=False)
    axs: npt.NDArray = field(init=False)
    CMAP_COLORS: tuple = ('copper', 'gray', 'twilight', 'summer')
    REGULAR_COLOR: str = 'copper'
    REGULAR_TITLE_COLOR: str = 'black'
    ERROR_COLOR: str = 'RdGy'
    ERROR_TITLE_COLOR: str = 'red'

    def __post_init__(self):
        self.jobs_n_columns: int = math.ceil(self.jobs ** 0.5)
        self.nodes_n_columns: int = math.ceil(self.nodes ** 0.5)

        jobs_n_rows: int = math.ceil(self.jobs/self.jobs_n_columns)
        nodes_n_rows: int = math.ceil(self.nodes/self.nodes_n_columns)

        n_rows: int = max(jobs_n_rows, nodes_n_rows)
        n_columns: int = self.nodes_n_columns + self.jobs_n_columns

        self.fig, self.axs = plt.subplots(n_rows, n_columns, figsize=(12, 6),)
        self.fig.patch.set_facecolor('white')

        self.axs = self.axs if len(self.axs.shape) > 1 else self.axs.reshape(1,-1)

        self._hide_unused(self.axs, nodes=self.nodes, jobs=self.jobs, nodes_n_columns=self.nodes_n_columns)

    @classmethod
    def _hide_unused(cls, axs: np.ndarray, nodes: int, jobs: int, nodes_n_columns: int):
        nodes_to_remove: Iterable[Axes] = axs[:,:nodes_n_columns].flatten()[nodes:]
        jobs_to_remove: Iterable[Axes] = axs[:, nodes_n_columns:].flatten()[jobs:]
        for ax in nodes_to_remove: plt.delaxes(ax)
        for ax in jobs_to_remove: plt.delaxes(ax)

    @classmethod
    def _draw(cls,matrix: np.ndarray,/,*, title: str, title_color: str ,ax: Axes, time: int, resource: int, cmap: str):
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100)
        ax.set_title(title, fontsize=10, color=title_color)
        ax.set_xticks(np.arange(0, time, 0.5), minor=True)
        ax.set_yticks(np.arange(0, resource, 0.5), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.grid(which='both', color='black', linestyle='-', linewidth=.5, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    @classmethod
    def _draw_job(cls, job: np.ndarray,/,*, title_color: str, idx: int, ax: Axes, time: int, resource: int, cmap: str, status: JobStatus):
        title: str = f"[J.{idx}]"
        cmap = cmap if cmap == cls.ERROR_COLOR else cls.CMAP_COLORS[status]
        cls._draw(job, title=title, title_color=title_color, ax=ax, time=time, resource=resource, cmap=cmap)
    @classmethod
    def _draw_node(cls, node: np.ndarray,/,*,title_color: str, idx: int, ax: Axes, time: int, resource: int, cmap: str):
        title: str = f"[N.{idx}]"
        cls._draw(node, title=title, title_color=title_color, ax=ax, time=time, resource=resource, cmap=cmap)

    def __call__(self, obs: dict[str, np.ndarray],/,*, current_time: int, error: None | tuple[int,int]) -> Any:
        self.fig.suptitle( f"Time: {current_time}", fontsize=16, fontweight='bold')
        nodes: npt.NDArray = obs['Usage']
        queue: npt.NDArray = obs['Queue']
        status: npt.NDArray[np.uint32] = obs['Status']
        cmap_color: Callable[[int,int],str] = lambda idx, pos: self.ERROR_COLOR if error and idx == error[pos] else self.REGULAR_COLOR
        title_color: Callable[[int,int],str] = lambda idx, pos: self.ERROR_TITLE_COLOR if error and idx == error[pos] else self.REGULAR_TITLE_COLOR
        node_ax: Callable[[int],npt.NDArray] = lambda n_idx: self.axs[n_idx // self.nodes_n_columns, n_idx % self.nodes_n_columns]
        job_ax: Callable[[int],npt.NDArray] = lambda j_idx: self.axs[j_idx // self.jobs_n_columns, self.nodes_n_columns + (j_idx % self.jobs_n_columns)]
        # update matries
        for n_idx, node in enumerate(nodes):
            self._draw_node(node, title_color=title_color(n_idx,0),idx=n_idx, ax=node_ax(n_idx), time=self.time, resource=self.resource, cmap=cmap_color(n_idx,0))
        for j_idx, job in enumerate(queue):
            self._draw_job(
                job,
                title_color=title_color(j_idx,1),
                idx=j_idx,
                ax=job_ax(j_idx),
                time=self.time,
                resource=self.resource,
                cmap=cmap_color(j_idx,1),
                status=JobStatus(status[j_idx])
            )
        # update figure
        plt.draw()
        plt.pause(self.cooldown)

@dataclass
class ClusterGenerator:
    nodes: int
    jobs: int
    resource: int
    time: int
    arrival: DistribConfig = field(default_factory=DEFUALT_ARRIVAL_FUNC)
    length: DistribConfig = field(default_factory=DEFUALT_LENGTH_FUNC)
    usage: DistribConfig = field(default_factory=DEFUALT_USAGE_FUNC)
    max_node_usage: float = field(default=255.0)

    def __call__(self, *args: Any, **kwds: Any) -> ClusterObject:
        logging.info(f"Generating Cluster with;  nodes: {self.nodes}, jobs: {self.jobs}, max node usage: {self.max_node_usage}")
        arrival_time: npt.NDArray[np.uint32] = (self.time * np.random.choice(self.arrival.options, size=(self.jobs), p=self.arrival.probability)).astype(np.uint32)
        job_length: npt.NDArray[np.int32] = 1 + self.time * np.random.choice(self.length.options, size=(self.jobs,self.resource), p=self.length.probability)
        usage: npt.NDArray[np.float64] = self.max_node_usage * np.random.choice(self.usage.options, size=(self.jobs), p=self.usage.probability)
        usage: npt.NDArray[np.float64] = np.tile(usage[..., np.newaxis, np.newaxis], (self.resource,self.time))
        mask = np.arange(usage.shape[-1]) >= job_length[..., np.newaxis]
        usage[mask] = .0
        jobs: Jobs = Jobs(arrival=arrival_time, usage=usage)
        nodes: npt.NDArray[np.float64] = np.full((self.nodes, self.resource, self.time), fill_value=self.max_node_usage, dtype=np.float64)
        return ClusterObject(
            nodes=nodes,
            jobs=jobs,
        )

@dataclass
class ClusterEnv(gym.Env):
    nodes: int
    jobs: int
    resource: int
    max_time: int
    # _render: Renderer = field(default_factory=Renderer)
    # _time: int = field(default=0)
    _cluster: ClusterObject = field(init=False)
    _logger: logging.Logger = field(init=False)
    _generator: ClusterGenerator = field(init=False)
    _renderer: ClusterRenderer = field(init=False)
    _action_error: tuple[int, int] | None = field(default=None)
    INNCORECT_ACTION_REWARD: int = field(default=-100)

    @property
    def time(self) -> int:
        return self._cluster.time

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._generator = ClusterGenerator(nodes=self.nodes,jobs=self.jobs,resource=self.resource, time=self.max_time)
        self._cluster = self._generator()
        self.observation_space = self._observation_space(self._cluster)
        self.action_space = self._action_space(self._cluster)
        self._renderer = ClusterRenderer(nodes=self.nodes, jobs=self.jobs, resource=self.resource, time=self.max_time)

    @classmethod
    def create_observation(cls, cluster: ClusterObject) -> dict:
        return dict(
            Usage=cluster.usage,
            Queue=cluster.queue,
            Nodes=cluster.nodes.copy(),
            Status=cluster.jobs_status.astype(np.intp)
        )
    @classmethod
    def _action_space(cls, cluster: ClusterObject) -> gym.spaces.Discrete:
        return gym.spaces.Discrete((cluster.n_nodes * cluster.n_jobs) + 1)
    @classmethod
    def _observation_space(cls, cluster: ClusterObject) -> gym.spaces.Dict:
        max_val = np.max(cluster.nodes)
        return gym.spaces.Dict(dict(
            Usage=gym.spaces.Box(
                low=0,
                high=max_val,
                shape=cluster.usage.shape,
                dtype=np.float64
            ),
            Queue=gym.spaces.Box(
                low=-1,
                high=max_val,
                shape=cluster.jobs.usage.shape,
                dtype=np.float64
            ),
            Nodes=gym.spaces.Box(
                low=0,
                high=max_val,
                shape=cluster.nodes.shape,
                dtype=np.float64
            ),
            Status=gym.spaces.Box(
                low=0,
                high=5,
                shape=cluster.jobs_status.shape,
                dtype=np.intp
            )
        ))


    @classmethod
    def _convert_index_to_space_action_idx(cls, cluster: ClusterObject, idx: int) -> tuple[int, int]:
        return  idx % cluster.n_nodes, idx // cluster.n_nodes

    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
            tick_action: bool = action == 0
            reward: float = 0
            self._action_error = None
            if tick_action:
                self._logger.info(f"Tick Cluster ...")
                self._cluster.tick()
            else:
                prefix: str = ""
                n_idx, j_idx = self._convert_index_to_space_action_idx(self._cluster, action-1)
                if not self._cluster.schedule(n_idx=n_idx,j_idx=j_idx):
                    self._action_error = (n_idx, j_idx)
                    prefix= "Can't"
                    reward += self.INNCORECT_ACTION_REWARD
                logging.info(f"{prefix} Allocating job {j_idx} into node {n_idx}")
            reward -= len(self._cluster.queue) / 2
            terminated: bool = self._cluster.all_jobs_complete()
            return self.create_observation(self._cluster), reward, terminated, False, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._cluster = self._generator()
        return self.create_observation(self._cluster), {}
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._renderer(
            self.create_observation(self._cluster),
            current_time=self.time,
            error=self._action_error
        )
