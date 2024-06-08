from returns.result import Success, Failure, Result
from typing import SupportsFloat, Any, List, Tuple
from pydantic import PositiveInt, validate_call
from dataclasses import dataclass, field
from gymnasium.core import RenderFrame
import gymnasium as gym
import logging
import torch

from .types import Status, ResourceError, StatusError, Action, ScheduleError


@dataclass
class Jobs:
    usage: torch.Tensor
    arrival_time: torch.Tensor
    status: Status = field(init=False)
    run_time: torch.Tensor = field(init=False)
    wait_time: torch.Tensor = field(init=False)
    length: torch.Tensor = field(init=False)

    @classmethod
    def caculate_time_untill_completion(cls, usage: torch.Tensor) -> torch.Tensor:
        len_by_resource: torch.Tensor = usage.shape[-1] - (
            torch.flip(usage, dims=[-1])
        ).argmax(dim=-1)
        return len_by_resource.max(axis=-1)

    def __post_init__(self):
        n_machines: PositiveInt = self.usage.shape[0]
        self.status = torch.full(fill_value=Status.Pending, size=(n_machines,), dtype=torch.int8)
        self.arrival_time = torch.ones(size=(n_machines,), dtype=torch.float32)
        self.run_time = torch.zeros(size=(n_machines,), dtype=torch.int32)
        self.wait_time = torch.zeros(size=(n_machines,), dtype=torch.int32)
        self.length = self.caculate_time_untill_completion(self.usage)

class ClusterEnvironment(gym.Env):
    SKIP_TIME_ACTION: PositiveInt = 0

    @validate_call
    def __init__(
        self,
        n_machines: PositiveInt,
        n_jobs: PositiveInt,
        n_resources: PositiveInt,
        time: PositiveInt,
    ) -> None:
        super(ClusterEnvironment, self).__init__()
        self.n_ticks: PositiveInt = 0
        self.n_machines = n_machines
        self.n_jobs = n_jobs
        self.n_resources = n_resources
        self.time = time
        self.machines: torch.Tensor = torch.full(
            (n_machines, n_resources, time), fill_value=3, dtype=torch.float32
        )
        self.jobs: torch.Tensor = torch.ones((n_jobs, n_resources, time), dtype=torch.float32)
        arrival_time = torch.ones((n_machines,) ,dtype=torch.int32)
        self.jobs: Jobs = Jobs(self.jobs, arrival_time=arrival_time)        


    @validate_call
    def convert_action(self, action: PositiveInt) -> Action:
        n_machines: PositiveInt = self.machines.shape[0]
        job_idx: PositiveInt = action % n_machines
        node_idx: PositiveInt = action // n_machines
        return (job_idx, node_idx)

    def schedule(self, m_idx: PositiveInt, j_idx: PositiveInt) -> Result[None, ScheduleError]:
        if self.jobs.status[m_idx] != Status.Pending:
            return Failure(StatusError)
        
        after_allocation: torch.Tensor = self.machines[m_idx] - self.jobs.usage[j_idx]
        if not torch.all(after_allocation >= 0):
            return Failure(ResourceError)
        
        self.machines[m_idx] -= self.jobs.usage[j_idx]
        self.jobs.status[j_idx] = Status.Running
        return Success(None)

    def observation(self) -> gym.Space:
        return dict(
            machines=self.machines,
            jobs=self.jobs.usage,
            status=self.jobs.status,
            time=self.n_ticks
        )
    
    def is_complete(self) -> bool:
        return all(self.jobs.status == Status.Complete) 
    
    def is_trunced(self) -> bool:
        return all(self.jobs.status != Status.Running)

    def step(self, action: int) -> Tuple[Any | SupportsFloat | bool | dict[str, Any]]:
        reward: float  = 0
        if action is self.SKIP_TIME_ACTION:
            logging.info(f"Time tick: {self.n_ticks}s -> {self.n_ticks+1}s")
            self.jobs.wait_time[self.jobs.status == Status.Pending] += 1
            self.jobs.run_time[self.jobs.status == Status.Running] += 1
            self.n_ticks += 1
            self.jobs.status[self.jobs.run_time == self.jobs.length] = Status.Complete
            self.jobs.status[self.jobs.arrival_time == self.n_ticks] = Status.Pending
            self.render()  # Update visualization
            return self.observation(), 0, self.is_trunced(), self.is_complete(), {}
        m_idx, j_idx = self.convert_action(action - 1)
        match self.schedule(m_idx, j_idx):
            case Success(None):
                logging.info(f"Allocating job '{j_idx}' to machine '{m_idx}'")
                reward += 1
            case Failure(StatusError):
                logging.warning(f"Can't allocate job: {j_idx} with status '{self.jobs.status[m_idx].name}'.")
                reward -= 1
            case Failure(ResourceError):
                logging.warning(f"Can't allocate job {j_idx} into {m_idx}, not enogh resource.")
                reward -= 1
            case _: 
                logging.error("Unexpected Error!")
                raise ValueError 
        return self.observation(), reward, self.is_trunced(), self.is_complete(), {}

    def reset(self, *, seed: PositiveInt | None = None, options: dict[str, Any] | None = None) -> Tuple[Any | dict[str, Any]]:
        self.n_ticks = 0
        self.jobs.status[:] = Status.NotArrived
        self.jobs.run_time[:] = 0
        self.jobs.wait_time[:] = 0
        self.jobs.status[self.jobs.arrival_time == 0] = Status.Pending
        return self.observation(), {}
    
    def render(self) -> RenderFrame | List[RenderFrame] | None:
        return None
        
    
