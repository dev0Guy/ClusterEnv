from typing import Any, Dict, SupportsFloat
from gymnasium.core import RenderFrame
from clusterenv.envs.base import JobStatus
from clusterenv.envs.cluster import ClusterEnv
from dataclasses import dataclass, field
import numpy.typing as npt
import gymnasium as gym
import numpy as np
import logging


@dataclass
class QueueWrapper(gym.Wrapper):
    """
    Object responsible of Organize Cluster Queue.
    TODO: FIX

    :param env: cluster gym.Envs
    :param limit: queue length
    :param mapper: inner attribute mapping original position of jobs to new one
    """

    env: gym.Env
    limit: int = field(default=3)
    mapper: npt.NDArray[np.intp] = field(init=False)

    def __post_init__(self):
        gym.Wrapper.__init__(self, self.env)
        # super(QueueWrapper, self).__init__(self.env)
        assert isinstance(self.unwrapped, ClusterEnv)
        # change original space
        self.observation_space["Queue"] = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=self.unwrapped.observation_space["Queue"].shape,
            dtype=np.float64,
        )
        self.mapper = np.arange(self.unwrapped.jobs)
        self.action_space = gym.spaces.Discrete(self.limit * self.unwrapped.nodes + 1)

    @classmethod
    def _convert_observation(
        cls,
        obs: Dict[str, npt.NDArray[np.float64]],
        *,
        mapper: npt.NDArray[np.intp],
        limit: None | int = None,
    ) -> tuple[npt.NDArray[np.intp], dict]:
        position = np.concatenate(
            [
                np.where(obs["Status"][mapper] == j_status.value)[0]
                for j_status in JobStatus
            ]
        )
        new_mapper: npt.NDArray[np.intp] = mapper[position]
        obs["Queue"] = obs["Queue"][new_mapper]
        obs["Status"] = obs["Status"][new_mapper]
        if limit:
            obs["Queue"] = obs["Queue"][:limit]
        return new_mapper, obs

    def step(self, action: int) -> tuple:
        if action != 0:
            n_idx, j_idx = self.unwrapped.convert_action(action - 1)
            action = self.unwrapped.convert_to_action(n_idx, self.mapper[j_idx]) + 1
        obs, *other = super().step(action)
        self.mapper, obs = self._convert_observation(
            obs, mapper=self.mapper, limit=self.limit
        )
        self.render()
        return obs, *other

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple:
        obs, *other = super().reset(seed=seed, options=options)
        self.mapper = np.arange(self.unwrapped.jobs)
        self.mapper, obs = self._convert_observation(
            obs, mapper=self.mapper, limit=self.limit
        )
        self.render()
        return obs, *other

    def render(self) -> RenderFrame | list[RenderFrame] | None | npt.NDArray[np.uint8]:
        _, obs = self._convert_observation(
            self.unwrapped._observation(self.unwrapped._cluster), mapper=self.mapper
        )
        if self.unwrapped._renderer:
            fig = self.unwrapped._renderer(
                obs,
                current_time=self.unwrapped.time,
                error=self.unwrapped._action_error,
            )
            if self.unwrapped.render_mode == "rgb_array":
                fig.canvas.draw()
                buf = fig.canvas.tostring_rgb()
                width, height = fig.canvas.get_width_height()
                return np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
