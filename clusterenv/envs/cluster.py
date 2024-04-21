from clusterenv.common import NodeGenerator, JobGenerator
from clusterenv.render import ClusterRenderer
from clusterenv._types import Jobs, Nodes, JobStatus
from gymnasium.core import ActType, ObsType, RenderFrame
from typing import List, Any, SupportsFloat, overload
import gymnasium as gym
import numpy as np
import logging


class ClusterEnv(gym.Env):
<<<<<<< Updated upstream
=======
    nodes: int
    jobs: int
    resource: int
    max_time: int
    cooldown: float = field(default=1.0)
    _cluster: ClusterObject = field(init=False)
    _logger: logging.Logger = field(init=False)
    _generator: ClusterGenerator = field(init=False)
    _renderer: ClusterRenderer = field(init=False)
    _action_error: tuple[int, int] | None = field(default=None)
    INNCORECT_ACTION_REWARD: int = field(default=-100)
    metadata: dict = field(default_factory=create_metadata)
    render_mode: str = field(default="rgb_array")
    @property
    def time(self) -> int:
        return self._cluster.time
>>>>>>> Stashed changes

    metadata = {"render_modes": ["human", "rgb_array", ''], "render_fps": 4}

    def _action_shape(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self.nodes) * len(self.jobs) + 1)

    def _observation_shape(self) -> gym.spaces.Dict:
        max_val: float = np.max(self.nodes.original)
        return gym.spaces.Dict(
            dict(
                Usage=gym.spaces.Box(
                    low=0, high=max_val, shape=self.jobs.usage.shape, dtype=float
                ),
                Queue=gym.spaces.Box(
                    low=-1,
                    high=max_val,
                    shape=self.jobs.usage.shape,
                    dtype=float,
                ),
                Nodes=gym.spaces.Box(
                    low=0, high=max_val, shape=self.nodes.original.shape, dtype=float
                ),
                Status=gym.spaces.Box(
                    low=0, high=5, shape=self.jobs.status.shape, dtype=float
                ),
            )
        )

    def __init__(
        self,
        n_nodes: int,
        n_jobs: int,
        n_resource: int,
        max_time: int,
        node_gen: NodeGenerator = NodeGenerator,
        job_gen: JobGenerator = JobGenerator,
        render_mode: str = '',
        cooldown: float = 0.05
    ):
        super(ClusterEnv, self).__init__()
        self._time = 0
        self._mapper = np.arange(n_jobs)
        self.render_mode = render_mode
        self._n_resource = n_resource
        self._max_time = max_time
        self._n_nodes, self._n_jobs = n_nodes, n_jobs
        self._node_gen, self._job_gen = node_gen, job_gen
        self._logger = logging.getLogger(self.__class__.__name__)
        self._plotter = ClusterRenderer(
            nodes=self._n_nodes,
            jobs=self._n_jobs,
            resource=self._n_resource,
            time=self._max_time,
            cooldown=cooldown,
        )
        self._action_error = None
        self._action_correct = None
        self.nodes: Nodes = self._node_gen.generate(
            self._n_nodes, self._n_resource, self._max_time, 255.0
        )
        self.jobs: Jobs = self._job_gen.generate(
            self._n_jobs, self._n_resource, self._max_time, 255.0
        )
        self.observation_space = self._observation_shape()
        self.action_space = self._action_shape()

    def _reorganize_observation(self) -> None:
        self._mapper = self._mapper[np.argsort(self.jobs.status[self._mapper], kind='stable')]
        self.jobs.reorganize(self._mapper)
        # TODO: Update jobs
        self._observation = dict(
            Usage=self.nodes.usage,
            Nodes=self.nodes.original,
            Status=self.jobs.status,
            Queue=self.jobs.usage,
        )

    def _tick(self) -> None:
        """Forward cluster time by tick.
        Updating running time of all running jobs by one, show new income jobs.
        """
        self._logger.info(f"Tick {self._time}->{self._time + 1}")
        self.jobs.inc(JobStatus.RUNNING)
        self._time += 1
        complete_jobs_idx: np.array = self.jobs.check_complete_jobs()
        arrived_jobs_idx: np.array = self.jobs.check_arrived_jobs(self._time)
        self._logger.info(f"Receive new jobs: {arrived_jobs_idx}")
        self._logger.info(f"Complete jobs: {complete_jobs_idx}")
        self.nodes.tick()

    def _schedule(self, n_idx: int, j_idx: int) -> bool:
        job_status, _, job_usage = self.jobs[j_idx]
        match job_status:
            case JobStatus.PENDING:
                free_n_space: np.ndarray = self.nodes.free_space(n_idx)
                if job_can_be_schedule := bool(np.all(free_n_space >= job_usage)):
                    self.nodes[n_idx] += job_usage
                    self.jobs[j_idx] = self.jobs[j_idx].change_status(JobStatus.RUNNING)
                    logging.info(f"Succeed Allocated j.{j_idx} to n.{n_idx}")
                else:
                    logging.info(f"Can't Allocate j.{j_idx} n.{n_idx}, not enough resource")
                return job_can_be_schedule
            case _:
                logging.info(f"Can't Allocate j.{j_idx} with status {JobStatus(job_status).name}")
                return False

    @overload
    def _convert_action(self, n_idx: int, j_idx: int) -> int: ...

    @overload
    def _convert_action(self, action: int) -> tuple[int, int]: ...

    def _convert_action(self, *args):
        n_nodes: int = len(self.nodes)
        if len(args) == 2 and all(isinstance(arg, int) for arg in args):
            n_idx, j_idx = args
            return j_idx * n_nodes + n_idx
        elif len(args) == 1 and isinstance(args[0], int):
            action = args[0]
            return action % n_nodes, action // n_nodes
        else:
            raise TypeError("Invalid arguments for convert_action method")

    def _has_terminate(self):
        return len(self.jobs.index_by_status(JobStatus.COMPLETE)) == len(self.jobs)

    def _has_additional_jobs(self):
        n_pending: int = len(self.jobs.index_by_status(JobStatus.COMPLETE))
        n_complete: int = len(self.jobs.index_by_status(JobStatus.RUNNING))
        return n_pending + n_complete != len(self.jobs)

    def _total_wait_time(self):
        return self.jobs.total_pending_time()

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        if self.render_mode:
            fig = self._plotter(
                self._observation,
                current_time=self._time,
                error=self._action_error,
                correct=self._action_correct,
            )
            if self.render_mode == "rgb_array":
                buf = fig.canvas.tostring_rgb()
                width, height = fig.canvas.get_width_height()
                expected_height = int(fig.get_figheight() * fig.dpi)
                expected_width = int(fig.get_figwidth() * fig.dpi)
                width_mult: int = expected_width // width
                height_mult: int = expected_height // height
                return np.frombuffer(buf, dtype=np.uint8).reshape(
                    (height_mult * height, width_mult * width, 3)
                )

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.nodes: Nodes = self._node_gen.generate(
            self._n_nodes, self._n_resource, self._max_time, 255.0
        )
        self.jobs: Jobs = self._job_gen.generate(
            self._n_jobs, self._n_resource, self._max_time, 255.0
        )
        self.jobs.check_arrived_jobs(self._time)
        self._reorganize_observation()
        self.render()
        return self._observation, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Make Cluster Step. Each step is a sub-time action, in other words player can pick inf action and time will not tick.
        Time will only tick when player take time tick action. Any other action is a scheduling action.
        When player take not possible action an error flag of self._action_error will be activated for wrapper Env (None).

        :param action: flatten action
        """
        info: dict = {}
        self._action_error = None
        self._action_correct = None
        self.render()
        if bool(action == 0):
            self._logger.info(f"Tick Cluster ...")
            self._tick()
        else:
            n_idx, j_idx = self._convert_action(int(action) - 1)
            if not self._schedule(n_idx, j_idx):
                self._action_error = (n_idx, j_idx)
            else:
                self._action_correct = (n_idx, j_idx)
        terminate: bool = self._has_terminate()
        truncated: bool = not self._has_additional_jobs()
        reward: float = -self._total_wait_time()
        self._reorganize_observation()
        self.render()
        return self._observation, reward, terminate, truncated, info
