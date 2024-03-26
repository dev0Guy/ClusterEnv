from typing import Iterable, ParamSpecArgs, Self, Any, SupportsFloat, Optional, Tuple
from dataclasses import dataclass, field
from gymnasium.core import RenderFrame
from typing_extensions import Callable, NamedTuple
from .base import ClusterObject, JobStatus, Jobs
from numpy._typing import NDArray
import numpy.typing as npt
import gymnasium as gym
import numpy as np
import logging
import math


@dataclass
class DistribConfig:
    options: list[Any]
    probability: list[float]


DEFUALT_ARRIVAL_FUNC: Callable = lambda: DistribConfig(
    options=[0.0, 0.2, 0.3], probability=[0.5, 0.4, 0.1]
)
DEFUALT_LENGTH_FUNC: Callable = lambda: DistribConfig(
    options=[0.0, 0.2, 0.3], probability=[0.7, 0.2, 0.1]
)
DEFUALT_USAGE_FUNC: Callable = lambda: DistribConfig(
    options=[0.1, 0.5, 1], probability=[0.7, 0.2, 0.1]
)


@dataclass
class ClusterGenerator:
    """
    Object responsible of generating/creating ClusterObject.

    :param nodes: number of nodes in cluster
    :param jobs: number of jobs in cluster
    :param resource: number of resource in cluster
    :param time: max job cluster length
    :param arrival: metadata to create jobs arrival rate
    :param length: metadata to create jobs length
    :param usage: metadata to create jobs uage
    :param max_node_usage: max usage allowed for all resource in cluster
    """

    nodes: int
    jobs: int
    resource: int
    time: int
    arrival: DistribConfig = field(default_factory=DEFUALT_ARRIVAL_FUNC)
    length: DistribConfig = field(default_factory=DEFUALT_LENGTH_FUNC)
    usage: DistribConfig = field(default_factory=DEFUALT_USAGE_FUNC)
    max_node_usage: float = field(default=255.0)

    def __call__(self) -> ClusterObject:
        """
        Generate cluster object usig class attributes.
        Creating Uniform jobs usage, represent the following: {Channel, Resource, Time}

        :return: cluster object, contain all nessiery information to represent cluster
        """
        logging.info(
            f"Generating Cluster with;  nodes: {self.nodes}, jobs: {self.jobs}, max node usage: {self.max_node_usage}"
        )
        arrival_time: npt.NDArray[np.uint32] = (
            self.time
            * np.random.choice(
                self.arrival.options, size=(self.jobs), p=self.arrival.probability
            )
        ).astype(np.uint32)
        job_length: npt.NDArray[np.int32] = 1 + self.time * np.random.choice(
            self.length.options,
            size=(self.jobs, self.resource),
            p=self.length.probability,
        )
        usage: npt.NDArray[np.float64] = self.max_node_usage * np.random.choice(
            self.usage.options, size=(self.jobs), p=self.usage.probability
        )
        usage: npt.NDArray[np.float64] = np.tile(
            usage[..., np.newaxis, np.newaxis], (self.resource, self.time)
        )
        mask = np.arange(usage.shape[-1]) >= job_length[..., np.newaxis]
        usage[mask] = 0.0
        jobs: Jobs = Jobs(arrival=arrival_time, usage=usage)
        nodes: npt.NDArray[np.float64] = np.full(
            (self.nodes, self.resource, self.time),
            fill_value=self.max_node_usage,
            dtype=np.float64,
        )
        return ClusterObject(
            nodes=nodes,
            jobs=jobs,
        )


@dataclass
class ClusterEnv(gym.Env):
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
    max_time: int
    cooldown: float = field(default=1.0)
    _cluster: ClusterObject = field(init=False)
    _logger: logging.Logger = field(init=False)
    _generator: ClusterGenerator = field(init=False)
    _renderer: Optional[Any] = field(init=False, default=None)
    _action_error: Optional[Tuple[int, int]] = field(default=None)
    INNCORECT_ACTION_REWARD: int = field(default=-100)
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode: Optional[str] = field(default=None)

    @property
    def time(self) -> int:
        """Current cluster time."""
        return self._cluster.time

    def __post_init__(self):
        super(ClusterEnv, self).__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._generator = ClusterGenerator(
            nodes=self.nodes, jobs=self.jobs, resource=self.resource, time=self.max_time
        )
        self._cluster: ClusterObject = self._generator()
        self.observation_space: gym.Space = self._observation_space(self._cluster)
        self.action_space: gym.Space = self._action_space(self._cluster)
        if self.render_mode in ("human", "rgb_array"):
            try:
                from clusterenv.envs.render import ClusterRenderer

                self._renderer = ClusterRenderer(
                    nodes=self.nodes,
                    jobs=self.jobs,
                    resource=self.resource,
                    time=self.max_time,
                    cooldown=self.cooldown,
                )
            except ImportError as e:
                print(e)
                raise ImportError(
                    "Render method use mathplotlib pleas install using 'pip install matplotlib'"
                )

    @classmethod
    def _mask_queue_observation(cls, cluster: ClusterObject):
        """
        Mask PENDING jobs.

        :param cluster: Cluster Object
        """
        obs: dict[str, npt.NDArray] = cls._observation(cluster)
        pendeing_jobs: npt.NDArray = cluster.jobs.status == JobStatus.PENDING
        obs["Queue"][~pendeing_jobs] = 0
        return obs

    @classmethod
    def _observation(cls, cluster: ClusterObject) -> dict:
        """
        Cluster Objservation

        :param cluster: cluster state

        :return: cluster usage;queue;nodes;job-status
        """

        return dict(
            Usage=cluster.usage,
            Queue=cluster.queue,
            Nodes=cluster.nodes.copy(),
            Status=cluster.jobs_status.astype(np.intp),
        )

    @classmethod
    def _action_space(cls, cluster: ClusterObject) -> gym.spaces.Discrete:
        """
        Env action space.

        :param cluster: cluster state

        :return: action space
        """
        return gym.spaces.Discrete((cluster.n_nodes * cluster.n_jobs) + 1)

    @classmethod
    def _observation_space(cls, cluster: ClusterObject) -> gym.spaces.Dict:
        """
        Env observation space.

        :param cluster: cluster state

        :return: observation space
        """
        max_val = np.max(cluster.nodes)
        return gym.spaces.Dict(
            dict(
                Usage=gym.spaces.Box(
                    low=0, high=max_val, shape=cluster.usage.shape, dtype=np.float64
                ),
                Queue=gym.spaces.Box(
                    low=-1,
                    high=max_val,
                    shape=cluster.jobs.usage.shape,
                    dtype=np.float64,
                ),
                Nodes=gym.spaces.Box(
                    low=0, high=max_val, shape=cluster.nodes.shape, dtype=np.float64
                ),
                Status=gym.spaces.Box(
                    low=0, high=5, shape=cluster.jobs_status.shape, dtype=np.intp
                ),
            )
        )

    def convert_action(self, idx: int) -> tuple[int, int]:
        """
        Convert 1D action into 2D Matrix action.

        :param idx: flatten action

        :return: matrix action idxes
        """
        return idx % self._cluster.n_nodes, idx // self._cluster.n_nodes

    def convert_to_action(self, n_idx: int, j_idx: int) -> int:
        """
        Convert 2D Matrix action into 1D action.

        :param n_idx: node index
        :param j_idx: job index

        :return: 2d matrix action
        """
        return j_idx * self._cluster.n_nodes + n_idx

    def step(
        self, action: int
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Make Cluster Step. Each step is a subtime action, in other words player can pick inf action and time will not tick.
        Time will only tick when player take time tick action. Any other action is a scheduling action.
        When player take not possible action an error flag of self._action_error will be activate for wrapper Env (None).

        :param action: flatten action

        :return: GYM step
        """
        reward: float = 0
        tick_action: bool = action == 0
        self._action_error = None
        if tick_action:
            self._logger.info(f"Tick Cluster ...")
            self._cluster.tick()
        else:
            prefix: str = ""
            n_idx, j_idx = self.convert_action(action - 1)
            if not self._cluster.schedule(n_idx=n_idx, j_idx=j_idx):
                self._action_error = (n_idx, j_idx)
                prefix = "Can't"
                reward += self.INNCORECT_ACTION_REWARD
            logging.info(f"{prefix} Allocating job {j_idx} into node {n_idx}")
        reward -= len(self._cluster.queue) / 2
        terminated: bool = self._cluster.all_jobs_complete()
        self.render()
        return (
            self._mask_queue_observation(self._cluster),
            reward,
            terminated,
            False,
            {},
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset Gym Env. Will create a new cluster object using Generator.

        :return: GYM step
        """
        self._cluster = self._generator()
        self.render()
        return self._mask_queue_observation(self._cluster), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Render Env using mathplotlib.
        """
        if self._renderer:
            if self.render_mode == "rgb_array":
                fig = self._renderer(
                    self._observation(self._cluster),
                    current_time=self.time,
                    error=self._action_error,
                )
                fig.canvas.draw()
                buf = fig.canvas.tostring_rgb()
                width, height = fig.canvas.get_width_height()
                return np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
