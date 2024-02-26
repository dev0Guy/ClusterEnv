from typing import ParamSpecArgs, Self, Any, SupportsFloat, Optional
from dataclasses import dataclass, field
from gymnasium.core import RenderFrame
from typing_extensions import Callable
from .base import ClusterObject, Jobs, Renderer
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

DEFUALT_ARRIVAL_FUNC: Callable = lambda: DistribConfig(options=[.0,.2,.3], probability=[.7,.2,.1])
DEFUALT_LENGTH_FUNC: Callable = lambda: DistribConfig(options=[.0,.2,.3], probability=[.7,.2,.1])
DEFUALT_USAGE_FUNC: Callable = lambda: DistribConfig(options=[0.1,0.5,1], probability=[.7,.2,.1])

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
    nodes: int = field(default=5)
    jobs: int = field(default=50)
    resource: int = field(default=3)
    max_time: int = field(default=10)
    # _render: Renderer = field(default_factory=Renderer)
    _time: int = field(default=0)
    _cluster: ClusterObject = field(init=False)
    _logger: logging.Logger = field(init=False)
    _generator: ClusterGenerator = field(init=False)
    INNCORECT_ACTION_REWARD: int = field(default=-100)
    @classmethod
    def create_observation(cls, cluster: ClusterObject) -> dict:
        return dict(
            Usage=cluster.usage,
            Queue=cluster.queue,
            Nodes=cluster.nodes.copy(),
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
            )
        ))


    @classmethod
    def _convert_index_to_space_action_idx(cls, cluster: ClusterObject, idx: int) -> tuple[int, int]:
        return  idx % cluster.n_nodes, idx // cluster.n_nodes
    @classmethod
    def render_obs(cls, obs: dict[str, np.ndarray],/,*, current_time: int,cooldown: int = 1) -> None:
        queue: np.ndarray = obs["Queue"]
        nodes: np.ndarray = obs["Usage"]

        n_nodes: int = len(nodes)
        n_jobs: int = len(queue)

        jobs_n_columns: int = math.ceil(n_jobs ** 0.5)
        jobs_n_rows: int = math.ceil(n_jobs/jobs_n_columns)

        nodes_n_columns: int = math.ceil(n_nodes ** 0.5)
        nodes_n_rows: int = math.ceil(n_nodes/nodes_n_columns)

        n_rows: int = max(jobs_n_rows, nodes_n_rows)
        n_columns: int = nodes_n_columns + jobs_n_columns

        fig, axs = plt.subplots(n_rows, n_columns, figsize=(12, 6), sharex=True, sharey=True)
        # title: str= f"Cluster: {current_time}"
        fig.suptitle( f"Cluster: {current_time}", fontsize=16)

        def draw(idx, r_idx: int , c_idx: int, matrix: np.ndarray, prefix: str):
            axs[r_idx, c_idx].imshow(matrix, cmap='gray', vmin=0, vmax=100)
            axs[r_idx, c_idx].set_title(f'{prefix} {idx+1}')
            axs[r_idx, c_idx].set_xlabel('Time')
            axs[r_idx, c_idx].set_ylabel('Resources')
            axs[r_idx, c_idx].set_xticks([])
            axs[r_idx, c_idx].set_yticks([])
            axs[r_idx, c_idx].grid(True, color='black', linewidth=0.5)

        for n_idx, node in enumerate(nodes):
            draw(
                idx=n_idx,
                r_idx=n_idx // nodes_n_columns,
                c_idx=n_idx % nodes_n_columns,
                matrix=node,
                prefix="Usage",
            )

        for j_id, job in enumerate(queue):
            draw(
                idx=j_id,
                r_idx=j_id // jobs_n_columns,
                c_idx=nodes_n_columns + (j_id % jobs_n_columns),
                matrix=job,
                prefix="Queue",
            )
        plt.show(block=False)
        plt.pause(cooldown)

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._generator = ClusterGenerator(nodes=self.nodes,jobs=self.jobs,resource=self.resource, time=self.max_time)
        self._cluster = self._generator()
        self.observation_space = self._observation_space(self._cluster)
        self.action_space = self._action_space(self._cluster)
    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
            tick_action: bool = action == 0
            reward: float = 0
            if tick_action:
                self._logger.info(f"Tick Cluster ...")
                self._cluster.tick()
            else:
                prefix: str = ""
                n_idx, j_idx = self._convert_index_to_space_action_idx(self._cluster, action-1)
                if not self._cluster.schedule(n_idx=n_idx,j_idx=j_idx):
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
        # return self._render.render_obs(self.create_observation(self._cluster))
         return self.render_obs(self.create_observation(self._cluster),current_time=self._time)
