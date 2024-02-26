from dataclasses import dataclass, field
from abc import ABC
from typing import Self, Optional, TypeVar
from typing_extensions import Type
import numpy.typing as npt
import numpy as np
import logging
import pygame
import enum
import math

class JobStatus(enum.IntEnum):
    """Enumeration representing the status of a job in a computing system."""
    NOT_ARRIVED = 0
    WAITTING = 1
    RUNNING = 2
    COMPLETE = 3

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
        _zero_idx: npt.NDArray[np.uint32] = (self.usage == 0).argmax(axis=-1)
        self._length = np.max(np.where(_zero_idx > 0,_zero_idx, len(self.usage[0])),axis=1)
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
        return self._usage.copy()
    @property
    def queue(self) -> npt.NDArray[np.float64]:
        # idx: npt.NDArray[np.bool_] = np.logical_or(self.jobs.status == JobStatus.WAITTING ,self.jobs.status == JobStatus.RUNNING)
        return self.jobs.usage.copy()
    def all_jobs_complete(self) -> bool:
        return bool(np.all(self.jobs.status == JobStatus.COMPLETE))
    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._run_time = np.zeros(len(self.jobs), dtype=np.uint32)
        self._usage  = np.zeros(self.nodes.shape)
        self.jobs.status[self._time == self.jobs.arrival] = JobStatus.WAITTING
        logging.info(f"Created cluster with; nodes: {self.n_nodes}, jobs: {self.n_jobs}")
        logging.info(f"Jobs Arrival Time: {self.jobs.arrival}")
        logging.info(f"Jobs Length Time: {self.jobs.length}")
    def tick(self):
        """Foward cluster time by one second."""
        self._run_time += (self.jobs.status == JobStatus.RUNNING).astype(np.uint32)
        self._time += 1
        # create index
        iteration_complete_jobs: npt.NDArray[np.bool_] = self._run_time == self.jobs.length
        arrived_jobs: npt.NDArray[np.bool_] = self.jobs.arrival == self._time
        logging.info(f"After tick time: {self._time}")
        # logging
        logging.info(f"Iteration completed jobs: {np.where(iteration_complete_jobs)}")
        logging.info(f"Iteration arrived jobs: {np.where(arrived_jobs)}")
        # set values
        self.jobs.status[iteration_complete_jobs] = JobStatus.COMPLETE
        self.jobs.usage[iteration_complete_jobs] = self._empty_job_cell_val
        self.jobs.status[arrived_jobs] = JobStatus.WAITTING
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
        logging.info(f"Try to schedule job {j_idx} with status {JobStatus(job_status).name}")
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

@dataclass
class Renderer:
    screen_width: int = field(default=800)
    screen_height: int = field(default=600)

    def __post_init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.font = pygame.font.SysFont('dsd', 24)

    def draw_matrix(self, matrix, x, y, title):
        cell_size = 40
        padding = 5
        font_height = 20
        rows, cols = matrix.shape
        text = self.font.render(title, True, (0, 0, 0))
        self.screen.blit(text, (x, y - font_height))
        for i in range(rows):
            for j in range(cols):
                color_value = int(255 * matrix[i, j] / 100)
                color = (min(255, max(0, color_value)),) * 3  # Ensure color values are within range
                pygame.draw.rect(self.screen, color, (x + j * cell_size, y + i * cell_size, cell_size, cell_size), 0)


    def render_obs(self, obs: dict[str, np.ndarray], cooldown: int = 1):
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

        self.screen.fill((255, 255, 255))

        for n_idx, node in enumerate(nodes):
            self.draw_matrix(node, n_idx % nodes_n_columns * 300, n_idx // nodes_n_columns * 300, f'Usage {n_idx+1}')

        for j_id, job in enumerate(queue):
            self.draw_matrix(job, (j_id % jobs_n_columns + nodes_n_columns) * 300, j_id // jobs_n_columns * 300, f'Queue {j_id+1}')

        pygame.display.flip()
        pygame.time.wait(cooldown * 1000)
