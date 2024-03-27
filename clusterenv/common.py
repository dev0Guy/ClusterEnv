from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from clusterenv._types import Jobs, JobStatus, Nodes
from typing import Protocol
import logging
import gymnasium as gym


class NodeGenerator(Protocol):
    @classmethod
    def generate(cls, n: int, r: int, t: int, max_usage: float) -> Nodes:
        nodes: npt.NDArray[float] = np.full(
            (n, r, t),
            fill_value=max_usage,
            dtype=float,
        )
        return Nodes(nodes)


class JobGenerator(Protocol):
    LENGTH = dict(option=[0.0, 0.2, 0.3], prob=[0.7, 0.2, 0.1])
    USAGE = dict(option=[1], prob=[1])
    ARRIVAL_TIME = dict(option=[0.0], prob=[1])

    @classmethod
    def generate(cls, n: int, r: int, t: int, max_usage: float) -> Jobs:
        submission: np.array = t * np.random.choice(
            cls.ARRIVAL_TIME["option"], size=n, p=cls.ARRIVAL_TIME["prob"]
        ).astype(int)
        length: np.array = 1 + t * np.random.choice(
            cls.LENGTH["option"], size=(n, r), p=cls.LENGTH["prob"]
        )
        usage: np.array = max_usage * np.random.choice(
            cls.USAGE["option"], size=n, p=cls.USAGE["prob"]
        ).astype(float)
        usage: np.ndarray = np.tile(usage[..., np.newaxis, np.newaxis], (r, t))
        mask = np.arange(usage.shape[-1]) >= length[..., np.newaxis]
        usage[mask] = 0.0
        return Jobs(usage=usage, submission=submission)


@dataclass(slots=True)
class Cluster:
    """
    Attributes
    ------
        nodes (Nodes): TODO
        jobs (Jobs): TODO

    """

    nodes: Nodes
    jobs: Jobs
    _logger: logging.Logger = field(init=False)
    _run_time: npt.NDArray[int] = field(init=False)
    _max_val: float = field(init=False)
    _time: int = 0

    @property
    def observation(self) -> dict:
        return dict(
            Usage=self.nodes.usage,
            Queue=self.jobs.usage,
            Nodes=self.nodes.original,
            Status=self.jobs.status,
        )

    @property
    def time(self) -> int:
        return self._time

    @property
    def max_usage(self) -> float:
        return self._max_val

    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.nodes) * len(self.jobs) + 1)

    def observation_shape(self) -> gym.Space:
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

    def has_terminate(self) -> bool:
        return len(self.jobs.index_by_status(JobStatus.COMPLETE)) == len(self.jobs)

    def has_additional_jobs(self) -> bool:
        n_pending: int = len(self.jobs.index_by_status(JobStatus.COMPLETE))
        n_complete: int = len(self.jobs.index_by_status(JobStatus.RUNNING))
        return n_pending + n_complete != len(self.jobs)

    def total_wait_time(self):
        return self.jobs.total_pending_time()

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._run_time = np.zeros(len(self.jobs)).astype(float)
        self._max_val = np.max(self.nodes)
        self.jobs.check_arrived_jobs(self._time)

    def tick(self) -> None:
        """Forward cluster time by tick.
        Updating running time of all running jobs by one, show new income jobs.
        """
        self._logger.info(f"Tick {self._time}->{self._time+1}")
        self.jobs.inc(JobStatus.RUNNING)
        self._time += 1
        complete_jobs_idx: npt.ArrayLike[int] = self.jobs.check_complete_jobs()
        arrived_jobs_idx: npt.ArrayLike[int] = self.jobs.check_arrived_jobs(self._time)
        self._logger.info(f"Receive new jobs: {arrived_jobs_idx}")
        self._logger.info(f"Complete jobs: {complete_jobs_idx}")
        self.nodes.tick()

    def schedule(self, n_idx: int, j_idx: int) -> bool:
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
