from dataclasses import dataclass, field
from abc import ABC
from typing import Self, Optional, TypeVar
from typing_extensions import Type
import numpy as np
import numpy.typing as npt
import logging
import enum

class JobStatus(enum.IntEnum):
    """Enumeration representing the status of a job in a computing system."""
    NOT_ARRIVED = enum.auto()
    WAITTING = enum.auto()
    RUNNING = enum.auto()
    COMPLETE = enum.auto()

@dataclass
class Jobs:
    """Represent Jobs in a computing system."""
    arrival: npt.NDArray[np.uint32] # time jobs arrive to the cluster
    usage: npt.NDArray[np.float64] # usage of job throw time
    _status: npt.NDArray[np.uint32] = field(init=False) # job status in cluster
    _length: npt.NDArray[np.uint32] = field(init=False) # length of each job
    @property
    def length(self) -> npt.NDArray[np.uint32]:
        return self._length
    @property
    def status(self) -> npt.NDArray[np.uint32]:
        return self._status
    def __post_init__(self):
        self._status = np.zeros(self.arrival.shape,dtype=JobStatus)
        self._length = np.zeros(self.arrival.shape, dtype=np.uint32)
    def __getitem__(self, idx: int) -> tuple[np.uint32, np.float64, np.uint32]:
        return self.arrival[idx], self.usage[idx], self._status[idx]
    def __setitem__(self, idx: int, status: JobStatus, usage_fill: float = -1):
        self._status[idx] = status
        self.usage[idx] = usage_fill
    def __len__(self) -> int:
        return len(self.arrival)



@dataclass
class ClusterObject:
    nodes: npt.NDArray[np.float64]
    jobs: Jobs
    _empty_job_cell_val: int = field(init=False, default=-1)
    _run_time: npt.NDArray[np.uint32] = field(init=False)
    _usage: npt.NDArray[np.float64] = field(init=False)
    _time: int = field(init=False,default=0)
    _logger: logging.Logger = field(init=False)
    @property
    def n_jobs(self) -> int:
        return len(self.jobs)
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)
    @property
    def usage(self) -> npt.NDArray[np.float64]:
        return self._usage
    @property
    def queue(self) -> npt.NDArray[np.float64]:
        idx: npt.NDArray[np.bool_] = np.logical_or(self.jobs.status == JobStatus.WAITTING ,self.jobs.status == JobStatus.RUNNING)
        print((self.jobs.status == JobStatus.WAITTING).shape,(self.jobs.status == JobStatus.RUNNING).shape)
        print(idx.shape, self.jobs.usage.shape)
        return self.jobs.usage[idx]
    def all_jobs_complete(self) -> bool:
        return bool(np.all(self.jobs.status == JobStatus.COMPLETE))
    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._run_time = np.zeros(len(self.jobs), dtype=np.uint32)
        self._usage  = np.zeros(self.nodes.shape)
        self.jobs.status[self._time == self.jobs.arrival] = JobStatus.WAITTING
    def tick(self):
        """Foward cluster time by one second."""
        self._time += 1
        self.jobs.status[self._run_time == self.jobs.length] = JobStatus.COMPLETE
        self.jobs.usage[self.jobs.status == JobStatus.COMPLETE] = self._empty_job_cell_val
        self.jobs.status[self._time == self.jobs.arrival] = JobStatus.WAITTING
        # move usage time by one
        self._usage = np.roll(self._usage, shift=-1,axis=-1)
        self._usage[:,:,-1] = 0

    def schedule(self, n_idx: int, j_idx: int) -> bool:
        """Schedule job on node n_idx if possible else return False.

        Args
        -----
            n_idx (int): cluster node index.
            j_idx (int): cluster job index.

        Returns
        -------
            bool: Able to schedule job to index.
        """
        _, job_usage, job_status = self.jobs[j_idx]
        match job_status:
            case JobStatus.WAITTING:
                job_can_be_schedule: bool = False
                node_free_space: npt.NDArray[np.float64] = self.nodes[n_idx] - self._usage[n_idx]
                if (job_can_be_schedule := bool(np.all(node_free_space >= job_usage))):
                    self._usage[n_idx] += job_usage
                    self.jobs[j_idx] = JobStatus.RUNNING
                return job_can_be_schedule
            case _:
                return False
