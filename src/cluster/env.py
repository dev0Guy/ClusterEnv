from returns.result import Success, Failure, Result
from typing import SupportsFloat, Any, List, Tuple, Optional
from pydantic import validate_call
from pydantic import PositiveInt, NonNegativeInt
from dataclasses import dataclass, field
from gymnasium.core import RenderFrame
import gymnasium as gym
import logging

import numpy as np
import numpy.typing as npt


from gymnasium.error import DependencyNotInstalled
from gymnasium import spaces

from .types import Status, ScheduleErrorType
from .types import MachineIndex, JobIndex, SkipTime


# Todo: change to numpy array , convert to type which are subsripable




@dataclass
class Jobs:
    usage: npt.NDArray
    arrival_time: npt.NDArray
    status: Status = field(init=False)
    run_time: npt.NDArray = field(init=False)
    wait_time: npt.NDArray = field(init=False)
    length: npt.NDArray = field(init=False)

    @classmethod
    def caculate_time_untill_completion(cls, usage: npt.NDArray) -> npt.NDArray:
        assert (
            len(usage.shape) == 3
        ), f"Usage should be a 3d matrix, not {len(usage.shape)}d"
        len_by_resource: npt.NDArray = usage.shape[-1] - np.argmax(
            np.flip(usage, axis=-1), axis=-1
        )
        return len_by_resource.max(axis=-1)

    def update_metrics(self):
        self.wait_time[self.status == Status.Pending] += 1
        self.run_time[self.status == Status.Running] += 1

    def update_status(self, n_ticks: NonNegativeInt):
        self.status[self.run_time == self.length] = Status.Complete
        self.status[self.arrival_time == n_ticks] = Status.Pending

    def __post_init__(self):
        n_machines: PositiveInt = self.usage.shape[0]
        self.status = np.full(
            fill_value=Status.Pending, shape=(n_machines,), dtype=np.int8
        )
        self.arrival_time = np.zeros(shape=(n_machines,), dtype=np.int32)
        self.run_time = np.zeros(shape=(n_machines,), dtype=np.int32)
        self.wait_time = np.zeros(shape=(n_machines,), dtype=np.int32)
        self.length = self.caculate_time_untill_completion(self.usage)


class ClusterEnvironment(gym.Env):
    SKIP_TIME_ACTION: int = 0
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    @validate_call
    def __init__(
        self,
        n_machines: PositiveInt,
        n_jobs: PositiveInt,
        n_resources: PositiveInt,
        time: PositiveInt,
        render_mode: Optional[str] = None,
    ) -> None:
        super(ClusterEnvironment, self).__init__()
        # Save arguments as attribute
        self.n_resources: PositiveInt = n_resources
        self.n_machines: PositiveInt = n_machines
        self.n_jobs: PositiveInt = n_jobs
        self.time: PositiveInt = time
        self.render_mode = render_mode
        # Initilize arguemts
        self.n_ticks: NonNegativeInt = 0
        self.machines: npt.NDArray = np.full(
            (n_machines, n_resources, time), fill_value=1, dtype=np.float32
        )
        self.jobs: npt.NDArray = np.ones((n_jobs, n_resources, time), dtype=np.float32)
        arrival_time: npt.NDArray = np.ones((n_machines,), dtype=np.int32)
        self.jobs: Jobs = Jobs(self.jobs, arrival_time=arrival_time)
        # TODO: check if works
        # self.action_space: gym.Space = gym.spaces.Discrete(start=0, n=1 + (self.n_jobs * self.n_machines))
        self.action_space: spaces.Space = spaces.Tuple(
            (
                spaces.Discrete(start=0, n=self.n_machines),  # for null action
                spaces.Discrete(start=0, n=self.n_jobs),
                spaces.Discrete(start=0, n=2),
            )
        )
        self.shape_space: gym.Space = gym.spaces.Dict(
            spaces=dict(
                machines=gym.spaces.Box(low=0, high=1, shape=self.machines.shape),
                jobs=gym.spaces.Box(low=0, high=1, shape=self.jobs.usage.shape),
                status=gym.spaces.Discrete(
                    start=int(Status.NotArrived), n=int(Status.Complete) + 1
                ),
                time=gym.spaces.Discrete(start=0, n=10_000),
            )
        )

    @validate_call
    def schedule(self, m_idx: MachineIndex, j_idx: JobIndex) -> Result[None, str]:
        if self.jobs.status[m_idx] != Status.Pending:
            return Failure(ScheduleErrorType.StatusError)

        after_allocation: npt.NDArray = self.machines[m_idx] - self.jobs.usage[j_idx]
        if not np.all(after_allocation >= 0):
            return Failure(ScheduleErrorType.ResourceError)

        self.machines[m_idx] -= self.jobs.usage[j_idx]
        self.jobs.status[j_idx] = Status.Running
        return Success(None)

    def observation(self) -> gym.Space:
        return dict(
            machines=self.machines,
            jobs=self.jobs.usage,
            status=self.jobs.status,
            time=self.n_ticks,
        )

    def is_complete(self) -> bool:
        return all(self.jobs.status == Status.Complete)

    def is_trunced(self) -> bool:
        return all(self.jobs.status != Status.Running)

    def machine_time_tick(self):
        """Roll usage by one & set new values as 1"""
        self.machines = np.roll(self.machines, -1, axis=-1)
        self.machines[:, :, -1] = 1

    @validate_call
    def step(
        self, action: Tuple[MachineIndex, JobIndex, SkipTime]
    ) -> Tuple[Any | SupportsFloat | bool | dict[str, Any]]:
        reward: float = 0
        m_idx, j_idx, skip_operation = action
        if skip_operation:
            logging.info(f"Time tick: {self.n_ticks}s -> {self.n_ticks+1}s")
            self.n_ticks +=  1
            self.jobs.update_metrics()
            self.jobs.update_status(self.n_ticks)
            self.machine_time_tick()
        else:
            match self.schedule(m_idx, j_idx):
                case Success(None):
                    logging.info(f"Allocating job '{j_idx}' to machine '{m_idx}'")
                    reward += 1
                case Failure(ScheduleErrorType.StatusError):
                    logging.warning(
                        f"Can't allocate job: {j_idx} with status '{Status(self.jobs.status[m_idx]).name}'."
                    )
                    reward -= 1
                case Failure(ScheduleErrorType.ResourceError):
                    logging.warning(
                        f"Can't allocate job {j_idx} into {m_idx}, not enogh resource."
                    )
                    reward -= 1
                case _:
                    logging.error("Unexpected Error!")
                    raise ValueError
        if self.render_mode == "human":
            self.render()
        return self.observation(), reward, self.is_trunced(), self.is_complete(), {}

    def reset(
        self, *, seed: PositiveInt | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Any | dict[str, Any]]:
        super(ClusterEnvironment, self).reset(seed=seed, options=options)
        # TODO: use seed
        self.n_ticks = 0
        self.jobs.status[:] = Status.NotArrived
        self.jobs.run_time[:] = 0
        self.jobs.wait_time[:] = 0
        self.jobs.status[self.jobs.arrival_time == 0] = Status.Pending
        self.machines = np.ones(self.machines.shape)
        if self.render_mode == "human":
            self.render()
        return self.observation(), {}

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                'e.g. gym("", render_mode="rgb_array")'
            )
            return
        # TODO: add visulization
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
        #     else:  # mode == "rgb_array"
        #         self.screen = pygame.Surface((self.screen_width, self.screen_height))
        # if self.clock is None:
        #     self.clock = pygame.time.Clock()
        return None
