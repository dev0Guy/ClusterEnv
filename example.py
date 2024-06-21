import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import time
import numpy.typing as npt


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

    def __init__(self, machines_shape: npt.ArrayLike, jobs_shape: npt.ArrayLike):
        assert (
            len(machines_shape) == len(jobs_shape) == 3
        ), "Jobs/Machine number of dim should be 3."
        self.extract_information_for_build(machines_shape, jobs_shape)
        pygame.init()
        self.title_font = pygame.font.Font(None, self._TIME_FONT_SIZE)
        self.font = pygame.font.Font(None, min(self.tile_size) // 4)
        self.screen = pygame.display.set_mode(self._SCREEN_SIZE)
        pygame.display.set_caption(self._TITLE)
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
            (self.tile_size[0] + self._CELL_SPACING) * machines_shape[2]
            - self._CELL_SPACING,
            (self.tile_size[1] + self._CELL_SPACING) * machines_shape[1]
            - self._CELL_SPACING,
        )
        self.slot_padding = (self.slot_size - _togther_size) // 2

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
                self.screen,
                self._SLOT_BACKGROUND_COLOR,
                (*spacing, *self.slot_size),
                self._SLOT_BORDER,
            )
            for r_idx in range(matrix.shape[0]):
                for c_idx in range(matrix.shape[1]):
                    cx_space = (
                        self.slot_padding[0]
                        + spacing[0]
                        + (self.tile_size[0] + self._CELL_SPACING) * c_idx
                    )
                    cy_space = (
                        self.slot_padding[1]
                        + spacing[1]
                        + (self.tile_size[1] + self._CELL_SPACING) * r_idx
                    )
                    rect = pygame.draw.rect(
                        self.screen,
                        self.get_color(matrix[r_idx, c_idx]),
                        (cx_space, cy_space, *self.tile_size),
                    )
                    text_surface = self.font.render(
                        f"{matrix[r_idx, c_idx]:.1f}", True, "black"
                    )
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)

    def draw(self, machines: npt.NDArray, jobs: npt.NDArray, time: int):
        self.screen.fill(self._BACKGROUND_WINDOW_COLOR)  # Clear the screen
        title_surface = self.title_font.render(f"Time: {time}", True, "white")
        title_rect = title_surface.get_rect(
            center=(
                self._SCREEN_SIZE[0] // 2,
                self._SCREEN_SIZE[1] - self._OUTER_SPACING[1],
            )
        )
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
        self.previous_machines = machines.copy()
        self.previous_jobs = jobs.copy()
        pygame.display.flip()


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.jobs = np.random.rand(7, 3, 8)
        self.machines = np.random.rand(7, 3, 8)
        self.visualizer = PyGameVisulizer(
            machines_shape=self.machines.shape, jobs_shape=self.jobs.shape
        )
        # Define action and observation space
        self.action_space = spaces.Discrete(10)  # Example action space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.jobs.shape, dtype=np.float32
        )

    def reset(self):
        self.jobs = np.random.rand(7, 3, 8)
        self.machines = np.random.rand(7, 3, 8)
        return self.jobs  # Return the initial observation

    def step(self, action):
        # Example step logic
        reward = np.random.rand()
        done = np.random.rand() > 0.95
        info = {}

        self.jobs = np.random.rand(7, 3, 8)
        self.machines = np.random.rand(7, 3, 8)

        return self.jobs, reward, done, info

    def render(self, mode="human"):
        import random

        self.visualizer.draw(self.machines, self.jobs, time=random.randint(0, 10))

    def close(self):
        pygame.quit()


# To run the environment
if __name__ == "__main__":
    env = CustomEnv()
    obs = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.1)  # Control the frame rate

    env.close()
