from returns.result import Success, Failure, Result
from typing import SupportsFloat, Any, List, Tuple, Optional
from pydantic import validate_call
from pydantic import PositiveInt
from dataclasses import dataclass, field
from gymnasium.core import RenderFrame
import gymnasium as gym
import logging

import numpy as np
import numpy.typing as npt


from gymnasium.error import DependencyNotInstalled
from gymnasium import spaces

from .types import Status, ScheduleErrorType
from .types import MachineIndex, JobIndex, SkipTime, ClusterTicks, ActionColor, Color
import pygame
import math
from enum import Enum
# TODO: visulize error of action
# TODO: visilize selection and time skip


class PyGameVisulizer:
    _BACKGROUND_WINDOW_COLOR: str = "#1E1E1E"
    _SLOT_SPACING: int = 10
    _SCREEN_SIZE: npt.ArrayLike = np.array((800, 600))
    _OUTER_SPACING: int = (15, 15)
    _TITLE: str = "Cluster Overview"
    _SLOT_BACKGROUND_COLOR: str = "#D9D9D9"
    _CELL_SPACING: int = 4
    _SLOT_BORDER: int = 5
    _TIME_FONT_SIZE: int = 20
    
    def __init__(self, machines_shape: npt.ArrayLike, jobs_shape: npt.ArrayLike, screen):
        assert (
            len(machines_shape) == len(jobs_shape) == 3
        ), "Jobs/Machine number of dim should be 3."
        self.extract_information_for_build(machines_shape, jobs_shape)
        self.title_font = pygame.font.Font(None,self._TIME_FONT_SIZE)
        self.font = pygame.font.Font(None, min(self.tile_size) // 4)
        self.screen = screen            
        # pygame.display.set_caption(self._TITLE)
        self.screen.fill(self._BACKGROUND_WINDOW_COLOR)
        self.previous_machines = np.zeros(machines_shape)
        self.previous_jobs = np.zeros(jobs_shape)

    def extract_information_for_build(
        self, machines_shape: npt.ArrayLike, jobs_shape: npt.ArrayLike
    ):
        n_machines_rows = math.ceil(math.sqrt(machines_shape[0]))
        n_jobs_rows = math.ceil(math.sqrt(jobs_shape[0]))
        self.n_machines_columns = math.ceil(machines_shape[0] / n_machines_rows)
        self.n_jobs_columns = math.ceil(jobs_shape[0] / n_jobs_rows)
        self.n_rows = max(n_jobs_rows, n_machines_rows)
        self.n_columns = self.n_machines_columns + self.n_jobs_columns
        self.inner_surface_size = (
            self._SCREEN_SIZE[0] - self._OUTER_SPACING[0] - self._TIME_FONT_SIZE,
            self._SCREEN_SIZE[1] - self._OUTER_SPACING[1] - self._TIME_FONT_SIZE,
        )
        self.slot_size = np.array(
            (
                math.floor(self.inner_surface_size[0] / (self.n_columns))
                - self._SLOT_SPACING,
                math.floor(self.inner_surface_size[1] / (self.n_rows))
                - self._SLOT_SPACING,
            )
        )
        self.slot_size[:] = np.min(self.slot_size)
        self.tile_size = (
            math.floor(self.slot_size[0] / machines_shape[2]) - self._CELL_SPACING,
            math.floor(self.slot_size[1] / machines_shape[1]) - self._CELL_SPACING,
        )
        _togther_size = (
            (self.tile_size[0] + self._CELL_SPACING) * machines_shape[2] - self._CELL_SPACING,
            (self.tile_size[1] + self._CELL_SPACING) * machines_shape[1] - self._CELL_SPACING
        )
        self.slot_padding = ( self.slot_size - _togther_size) // 2

    @staticmethod
    def interpolate_color(color1, color2, factor):
        result = []
        for i in range(3):
            result.append(int(color1[i] + (color2[i] - color1[i]) * factor))
        return tuple(result)

    @classmethod
    def get_color(cls, value: float):
        value = max(0, min(1, value))
        color1 = (231, 76, 60)
        color2 = (241, 196, 15)
        color3 = (26, 188, 156)
        if value < 0.5:
            return cls.interpolate_color(color1, color2, value * 2)
        else:
            return cls.interpolate_color(color2, color3, (value - 0.5) * 2)

    def draw_cells(self, spacing: Tuple, matrix: float):
        for r_idx in range(matrix.shape[0]):
            for c_idx in range(matrix.shape[1]):
                value = matrix[r_idx, c_idx]
                cx_space = (
                    self.slot_padding[0] + spacing[0]
                    + (self.tile_size[0] + self._CELL_SPACING) * c_idx
                )
                cy_space = (
                    self.slot_padding[1] + spacing[1]
                    + (self.tile_size[1] + self._CELL_SPACING) * r_idx
                )
                rect = pygame.draw.rect(
                    self.screen,
                    self.get_color(value),
                    (cx_space, cy_space, *self.tile_size),
                )
                text_surface = self.font.render(
                    f"{value:.1f}", True, "black"
                )
                text_rect = text_surface.get_rect(center=rect.center)
                self.screen.blit(text_surface, text_rect)

    def draw_single(
        self,
        current_matrices: npt.NDArray,
        previous_matrices: npt.NDArray,
        *,
        start_column: int,
        column_length: int,
    ):
        for idx, matrix in enumerate(current_matrices):
            r_idx = idx // column_length
            c_idx = start_column + idx % column_length
            spacing = self.slot_size + self._SLOT_SPACING
            spacing[0] *= c_idx
            spacing[1] *= r_idx
            spacing = self._OUTER_SPACING + spacing
            pygame.draw.rect(
                self.screen, self._SLOT_BACKGROUND_COLOR, (*spacing, *self.slot_size)
            )
            pygame.draw.rect(
                self.screen, self._SLOT_BACKGROUND_COLOR, (*spacing, *self.slot_size), self._SLOT_BORDER
            )
            self.draw_cells(spacing=spacing, matrix=matrix)

    def draw_single_slot(self, r_idx: int , c_idx: int, color: str):
        spacing = self.slot_size + self._SLOT_SPACING
        spacing[0] *= c_idx
        spacing[1] *= r_idx
        spacing +=  self._OUTER_SPACING
        pygame.draw.rect(
            self.screen, color, (*spacing, *self.slot_size)
        )
        pygame.draw.rect(
            self.screen, color, (*spacing, *self.slot_size), self._SLOT_BORDER
        )
        return spacing


    def draw(self, machines: npt.NDArray, jobs: npt.NDArray, time: int, color: ActionColor):
        self.screen.fill(self._BACKGROUND_WINDOW_COLOR)
        title_surface = self.title_font.render(f"Time: {time}", True, "white")
        title_rect = title_surface.get_rect(center=(self._SCREEN_SIZE[0] // 2, self._SCREEN_SIZE[1] - self._OUTER_SPACING[1]))
        self.screen.blit(title_surface, title_rect)
        self.draw_single(
            machines,
            self.previous_machines,
            start_column=0,
            column_length=self.n_machines_columns,
        )
        self.draw_single(
            jobs,
            self.previous_jobs,
            start_column=self.n_machines_columns,
            column_length=self.n_jobs_columns,
        )
        if color:
            (m_idx, j_idx), color = color
            r_idx, c_idx = m_idx // self.n_machines_columns, 0 + m_idx % self.n_machines_columns
            spacing = self.draw_single_slot(r_idx=r_idx, c_idx=c_idx, color=color)
            self.draw_cells(matrix=machines[m_idx], spacing=spacing)
            r_idx, c_idx = j_idx // self.n_jobs_columns, self.n_machines_columns + j_idx % self.n_jobs_columns
            spacing = self.draw_single_slot(r_idx=r_idx, c_idx=c_idx, color=color)
            self.draw_cells(matrix=jobs[j_idx], spacing=spacing)
        self.previous_machines = machines.copy()
        self.previous_jobs = jobs.copy()

@dataclass
class Jobs:
    usage: npt.NDArray
    arrival_time: npt.NDArray
    status: npt.NDArray = field(init=False)
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

    @validate_call
    def update_metrics(self):
        self.wait_time[self.status == Status.Pending] += 1
        self.run_time[self.status == Status.Running] += 1

    @validate_call
    def update_status(self, n_ticks: ClusterTicks):
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
        self.n_ticks: ClusterTicks = 0
        self.machines: npt.NDArray = np.full(
            (n_machines, n_resources, time), fill_value=1, dtype=np.float32
        )
        self.jobs: npt.NDArray = np.ones((n_jobs, n_resources, time), dtype=np.float32)
        arrival_time: npt.NDArray = np.ones((n_machines,), dtype=np.int32)
        self.jobs: Jobs = Jobs(self.jobs, arrival_time=arrival_time)
        self.visulizer = None
        self.clock = None
        self.action_color: ActionColor = None
        # Initilize spaces
        self.action_space: spaces.Space = spaces.Tuple(
            (
                spaces.Discrete(start=0, n=self.n_machines),
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
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward: float = 0
        m_idx, j_idx, skip_operation = action
        if skip_operation:
            logging.info(f"Time tick: {self.n_ticks}s -> {self.n_ticks+1}s")
            self.n_ticks += 1
            self.jobs.update_metrics()
            self.jobs.update_status(self.n_ticks)
            self.machine_time_tick()
            self.action_color = None
        else:
            match self.schedule(m_idx, j_idx):
                case Success(None):
                    self.action_color = ((m_idx, j_idx), Color.Correct)
                    logging.info(f"Allocating job '{j_idx}' to machine '{m_idx}'")
                    reward += 1
                case Failure(ScheduleErrorType.StatusError):
                    self.action_color = ((m_idx, j_idx), Color.InCorrect)
                    logging.warning(
                        f"Can't allocate job: {j_idx} with status '{Status(self.jobs.status[m_idx]).name}'."
                    )
                    reward -= 1
                case Failure(ScheduleErrorType.ResourceError):
                    self.action_color = ((m_idx, j_idx), Color.InCorrect)
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
        self.n_ticks = 0
        self.jobs.status[:] = Status.NotArrived
        self.jobs.run_time[:] = 0
        self.jobs.wait_time[:] = 0
        self.jobs.status[self.jobs.arrival_time == 0] = Status.Pending
        self.machines = np.ones(self.machines.shape)
        if self.render_mode == "human":
            self.render()
        return self.observation(), {}

    def close(self):
        pygame.quit()

    def render(self, mode='human') -> RenderFrame | List[RenderFrame] | None:
        if self.render_mode is None:
            if self.spec:
                gym.logger.warn(
                    "You are calling render method without specifying any render mode. "
                    "You can specify the render_mode at initialization, "
                    f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
                )
            return
        
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e
        if self.visulizer is None:
            pygame.init()
            if self.render_mode == 'human':
                screen =  pygame.display.set_mode(PyGameVisulizer._SCREEN_SIZE)
            else:
                screen = pygame.Surface(PyGameVisulizer._SCREEN_SIZE)
            self.visulizer = PyGameVisulizer(machines_shape=self.machines.shape, jobs_shape=self.jobs.usage.shape, screen=screen)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.visulizer.draw(self.machines, self.jobs.usage, self.n_ticks, color=self.action_color)

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.visulizer.screen)), axes=(1, 0, 2)
            )