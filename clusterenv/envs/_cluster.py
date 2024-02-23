from dataclasses import dataclass, field
from .base import ClusterObject, Jobs



from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Any, SupportsFloat
from .base import *
import gymnasium as gym
import numpy as np
import logging
import math

@dataclass
class ClusterEnv(gym.Env):
    n_nodes: int
    n_jobs: int
    n_resources: int
    time: int
    in_correct_action_reward: int = field(init=False, default=-10)

    @staticmethod
    def _create_cluster(*,n_nodes: int, n_jobs: int, n_resources: int, time: int):
        jobs: Jobs = Jobs.generate(
            (n_jobs, n_resources, time),
            length=LengthConfig(options=[1],probability=[1]),
            load=LoadConfig(options=[100],probability=[1]),
        )
        nodes: np.array = np.full((n_resources, time),100)
        nodes: np.ndarray = np.tile(nodes[...,np.newaxis],(n_nodes)).transpose(2,0,1)
        return Cluster(nodes=nodes,jobs=jobs)

    @staticmethod
    def convert(*, n_nodes: int, action: int):
        return  action % n_nodes, action // n_nodes

    @property
    def _observation(self) -> dict:
        return dict(
            Usage=self._cluster.usage,
            Queue=self._cluster.queue,
            Nodes=self._cluster.nodes.copy(),
        )

    def __post_init__(self):
        super(ClusterEnv, self).__init__()
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._cluster: Cluster = self._create_cluster(
            n_nodes=self.n_nodes,
            n_jobs=self.n_jobs,
            n_resources=self.n_resources,
            time=self.time
        )
        self.action_space = gym.spaces.Discrete((self.n_nodes * self.n_jobs) + 1) # add null action
        max_val = np.max(self._cluster.nodes)
        self.observation_space = gym.spaces.Dict(dict(
            Usage=gym.spaces.Box(
                low=0,
                high=max_val,
                shape=self._cluster.usage.shape,
                dtype=np.float64
            ),
            Queue=gym.spaces.Box(
                low=-1,
                high=max_val,
                shape=self._cluster.queue.shape,
                dtype=np.float64
            ),
            Nodes=gym.spaces.Box(
                low=0,
                high=max_val,
                shape=self._cluster.queue.shape,
                dtype=np.float64
            )
        ))

    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        tick_action: bool = action == 0
        reward: int = 0
        if tick_action:
            self._logger.debug(f"Tick Cluster ...")
            self._cluster.tick()
        else:
            prefix: str = ""
            n_idx, j_idx = self.convert(n_nodes=self.n_nodes, action=action-1)
            if not self._cluster.schedule(n_idx=n_idx,j_idx=j_idx):
                prefix= "Can't"
                reward += self.in_correct_action_reward
            logging.debug(f"{prefix} Allocating job {j_idx} into node {n_idx}")
        reward -= len(self._cluster.queue) / 2
        terminated: bool = np.all(self._cluster.jobs.status == JobStatus.COMPLETE)
        return self._observation, reward, terminated, False, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._cluster: Cluster = self._create_cluster(
            n_nodes=self.n_nodes,
            n_jobs=self.n_jobs,
            n_resources=self.n_resources,
            time=self.time
        )
        return self._observation, {}

    @classmethod
    def render_obs(cls, obs: dict[str, np.ndarray]) -> None:
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

        _, axs = plt.subplots(n_rows, n_columns, figsize=(12, 6), sharex=True, sharey=True)

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
        plt.pause(1)

    def render(self):
        """
        Render the current state of the environment.
        This method visualizes the node usage with borders and shows a different matrix for each node with a color range.
        """
        return self.render_obs(self._observation)
