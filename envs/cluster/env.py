import logging
from dataclasses import dataclass, field
from typing import Literal, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.spaces import Box, Dict, MultiDiscrete
from enum import IntEnum, StrEnum
from returns.result import Result, Failure, Success
from collections import defaultdict
import numpy.typing as npt

MachineIndex = int
JobIndex = int


class JobStatus(IntEnum):
    Null = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


class ScheduleErrorType(StrEnum):
    ResourceError: str = "Not Enogh resource int machine"
    StatusError: str = 'Scheduling a job can only be with "pending" status.'
    SkipTime: str = "Skipping job due to time limit."


@dataclass
class ClusterEnv(gym.Env):

    n_machines: int
    n_jobs: int
    n_resource: int
    max_ticks: int
    render_mode: Literal["human", "rgb_array", None] = field(kw_only=True, default=None)
    machine_availability: np.ndarray = field(init=False)
    jobs_usage: np.ndarray = field(init=False)
    jobs_status: np.ndarray = field(init=False)
    jobs_length: np.ndarray = field(init=False)
    jobs_arrival: np.ndarray = field(init=False)
    job_metrics: dict = field(init=False)
    n_ticks: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        super(ClusterEnv, self).__init__()
        self.observation_space = Dict(
            {
                "machinesAvailability": Box(
                    0,
                    1,
                    (self.n_machines, self.n_resource, self.max_ticks),
                    dtype=float,
                ),
                "jobsUsage": Box(0, 1, (self.n_jobs, self.n_resource, self.max_ticks), dtype=float),
                "jobStatus": MultiDiscrete([max(JobStatus)] * self.n_jobs, dtype=float),
                "ticks": Box(0, np.inf, (1,), dtype=int),
            }
        )
        self.action_space = MultiDiscrete([2, self.n_machines, self.n_jobs], dtype=int)
        self.job_metrics = {
            "idle": np.zeros((self.n_jobs,), dtype=int),
            "run": np.zeros((self.n_jobs,), dtype=int),
        }

    def tick_time(self):
        self.n_ticks += 1
        # # update run time
        self.update_job_metrics()
        # # shit running jobs one tick forward
        self.machine_availability = np.roll(self.machine_availability, -1, axis=-1)
        self.machine_availability[:, :, -1] = 1

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        scheduled: Result = Failure(ScheduleErrorType.SkipTime)
        should_tick, m_idx, j_idx = action
        if not should_tick:
            scheduled = self.schedule(m_idx, j_idx)
        match scheduled:
            case Success(_):
                pass
            case Failure(ScheduleErrorType.SkipTime):
                self.tick_time()
            case Failure(ScheduleErrorType.StatusError | ScheduleErrorType.ResourceError) as e:
                self.tick_time()
            case Failure(_):
                raise ValueError
        idle = self.job_metrics["idle"]
        reward: float = (-1 / idle[idle != 0]).sum()
        n_running: int = (self.jobs_status == int(JobStatus.RUNNING)).sum()
        n_done: int = (self.jobs_status == int(JobStatus.COMPLETED)).sum()
        done: bool = n_done == self.n_jobs or n_running + n_done == self.n_jobs
        return self.get_observation(), reward, False, done, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.n_ticks = 0
        # TODO: change to random
        self.machine_availability = np.full((self.n_machines, self.n_resource, self.max_ticks), 1.0, dtype=float)
        self.jobs_usage = np.full((self.n_jobs, self.n_resource, self.max_ticks), 1.0, dtype=float)  # TODO: implement
        self.jobs_status = np.full((self.n_jobs,), JobStatus.PENDING, dtype=int)
        self.jobs_length = np.full((self.n_jobs,), self.max_ticks, dtype=int)
        self.jobs_arrival = np.full((self.n_jobs,), 0, dtype=int)
        self.job_metrics = {
            "idle": np.zeros((self.n_jobs,), dtype=int),
            "run": np.zeros((self.n_jobs,), dtype=int),
        }
        return self.get_observation(), {}

    def get_observation(self) -> dict:
        """Return the current cluster observation"""
        obs = {
            "machinesAvailability": self.machine_availability,
            "jobsUsage": self.jobs_usage,
            "jobStatus": self.jobs_status,
            "ticks": self.n_ticks,
        }
        return obs

    def close(self):
        super(self.__class__, self).close()
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()

    def schedule(
        self,
        m_idx: MachineIndex,
        j_idx: JobIndex,
    ) -> Result[None, ScheduleErrorType]:
        """Schedule Jobs in ``j_idx`` into  ``m_idx``.

        When Machine  ``m_idx`` can run & schedule job ``j_idx`` Success will be returned.
        Otherwise, Failure will be return with error message.

        Args:
            m_idx (MachineIndex): positive integer index of machine in ``self.machine_availability``.
            j_idx (JobIndex): positive integer index of job in ``self.jobs_usage``.

        Returns:
            Result[None, ScheduleErrorType]
        """
        if self.jobs_status[j_idx] != JobStatus.PENDING:
            return Failure(ScheduleErrorType.StatusError)
        allocated: np.ndarray = self.machine_availability[m_idx] - self.jobs_usage[j_idx]
        if not np.all(allocated >= 0):
            return Failure(ScheduleErrorType.ResourceError)
        self.machine_availability[m_idx] -= self.jobs_usage[j_idx]
        self.jobs_status[j_idx] = JobStatus.RUNNING
        return Success(None)

    def update_job_metrics(self) -> None:
        """Run after time tick. Ensure metrics \ status are updated accordingly"""
        self.job_metrics["idle"][self.jobs_status == JobStatus.PENDING] += 1
        self.job_metrics["run"][self.jobs_status == JobStatus.RUNNING] += 1
        self.jobs_status[self.jobs_arrival == self.n_ticks] = JobStatus.PENDING
        self.jobs_status[self.job_metrics["run"] == self.jobs_length] = JobStatus.COMPLETED
