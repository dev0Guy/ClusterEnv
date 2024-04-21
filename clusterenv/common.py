from numba import float64, int32, njit, jit, deferred_type
from numba.experimental import jitclass
import numpy as np
import gymnasium as gym
from clusterenv._types import JobStatus
from typing import Protocol, Dict
from clusterenv._types import Nodes, Jobs
import logging


@njit
def _node_generate(n: int, r: int, t: int, max_usage: float) -> np.ndarray:
    nodes: np.array = np.full(
        (n, r, t),
        fill_value=max_usage,
        dtype=float,
    )
    return nodes


def _job_generate(n: int, r: int, t: int, max_usage: float, arrival: Dict[str, float], length: Dict[str, float], usage: Dict[str, float]) -> tuple[np.ndarray, np.array]:
    submission: np.array = t * np.random.choice(
        arrival["option"], size=n, p=arrival["prob"]
    ).astype(int)
    length: np.array = 1 + t * np.random.choice(
        length["option"], size=(n, r), p=length["prob"]
    )
    usage: np.array = max_usage * np.random.choice(
        usage["option"], size=n, p=usage["prob"]
    ).astype(float)
    usage: np.ndarray = np.tile(usage[..., np.newaxis, np.newaxis], (r, t))
    mask = np.arange(usage.shape[-1]) >= length[..., np.newaxis]
    usage[mask] = 0.0
    return usage, submission


class NodeGenerator(Protocol):

    @classmethod
    def generate(cls, n: int, r: int, t: int, max_usage: float) -> Nodes:
        return Nodes(_node_generate(n, r, t, max_usage))


class JobGenerator(Protocol):
    LENGTH = dict(option=[0.0, 0.2, 0.3], prob=[0.7, 0.2, 0.1])
    USAGE = dict(option=[1], prob=[1])
    ARRIVAL_TIME = dict(option=[0.0], prob=[1])

    @classmethod
    def generate(cls, n: int, r: int, t: int, max_usage: float) -> Jobs:
        return Jobs(*_job_generate(n, r, t, max_usage, length=cls.LENGTH, arrival=cls.ARRIVAL_TIME, usage=cls.USAGE))


spec = [
    ('nodes', float64[:, :, :]),
    ('jobs', float64[:, :, :]),
    ('_logger', float64),
    ('_run_time', float64[:]),
    ('_max_val', float64),
    ('_time', int32)
]

@jitclass(spec)
class Cluster:

    def __init__(self, nodes, jobs):
        self.nodes = nodes
        self.jobs = jobs
        self._logger = logging.getLogger(self.__class__.__name__)
        self._run_time = np.zeros(len(self.jobs)).astype(float)
        self._max_val = np.max(self.nodes)
        self.jobs.check_arrived_jobs(self._time)

    @property
    def observation(self):
        return dict(
            Usage=self.nodes.usage,
            Queue=self.jobs.usage,
            Nodes=self.nodes.original,
            Status=self.jobs.status,
        )

    @property
    def time(self):
        return self._time

    @property
    def max_usage(self):
        return self._max_val

    def action_space(self):
        return gym.spaces.Discrete(len(self.nodes) * len(self.jobs) + 1)

    def observation_shape(self):
        max_val = np.max(self.nodes.original)
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

    def has_terminate(self):
        return len(self.jobs.index_by_status(JobStatus.COMPLETE)) == len(self.jobs)

    def has_additional_jobs(self):
        n_pending = len(self.jobs.index_by_status(JobStatus.COMPLETE))
        n_complete = len(self.jobs.index_by_status(JobStatus.RUNNING))
        return n_pending + n_complete != len(self.jobs)

    def total_wait_time(self):
        return self.jobs.total_pending_time()

    def tick(self):
        self._logger.info(f"Tick {self._time}->{self._time+1}")
        self.jobs.inc(JobStatus.RUNNING)
        self._time += 1
        complete_jobs_idx = self.jobs.check_complete_jobs()
        arrived_jobs_idx = self.jobs.check_arrived_jobs(self._time)
        self._logger.info(f"Receive new jobs: {arrived_jobs_idx}")
        self._logger.info(f"Complete jobs: {complete_jobs_idx}")
        self.nodes.tick()

    def schedule(self, n_idx, j_idx):
        job_status, _, job_usage = self.jobs[j_idx]
        if job_status == JobStatus.PENDING:
            free_n_space = self.nodes.free_space(n_idx)
            job_can_be_schedule = np.all(free_n_space >= job_usage)
            if job_can_be_schedule:
                self.nodes[n_idx] += job_usage
                self.jobs[j_idx] = self.jobs[j_idx].change_status(JobStatus.RUNNING)
                logging.debug(f"Succeed Allocated j.{j_idx} to n.{n_idx}")
            else:
                logging.debug(f"Can't Allocate j.{j_idx} n.{n_idx}, not enough resource")
            return job_can_be_schedule
        else:
            logging.debug(f"Can't Allocate j.{j_idx} with status {JobStatus(job_status).name}")
            return False
