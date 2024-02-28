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
    env: gym.Env
    limit: int = field(default=3)
    mapper: npt.NDArray[np.intp] = field(init=False)

    def __post_init__(self):
        super(QueueWrapper, self).__init__(self.env)
        assert isinstance(self.unwrapped, ClusterEnv)
        self.unwrapped.observation_space["Queue"]= gym.spaces.Box(
            low=-1,
            high=np.inf,
            shape=(self.limit, *self.unwrapped.observation_space["Queue"].shape[1:]),
            dtype=np.float64
        )
        self.mapper = np.arange(self.unwrapped.jobs)
        self.action_space = gym.spaces.Discrete(self.limit * self.unwrapped.nodes +1)

    @classmethod
    def _create_mapper(cls, status: npt.NDArray[np.uint32], mapper: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        return np.concatenate([np.where(j_status == status[mapper])[0] for j_status in JobStatus])

    @classmethod
    def _transpose_observation(cls, obs: Dict[str, npt.NDArray], mapper: npt.NDArray[np.intp]) -> Dict[str, npt.NDArray]:
        prev_mapper, next_mapper = mapper, cls._create_mapper(obs["Status"], mapper)
        obs["Queue"] = obs["Queue"][prev_mapper][next_mapper]
        obs["Status"] = obs["Status"][prev_mapper][next_mapper]
        return obs



    @classmethod
    def _observation_limit(cls, obs: Dict[str, npt.NDArray], *, limit: int) -> Dict[str, npt.NDArray]:
        obs["Queue"] = obs["Queue"][:limit]
        obs["Status"] = obs["Status"][:limit]
        return obs


    def step(self, action: int) -> tuple:
        obs, *other = super().step(action)
        new_obs = self._observation_limit(self._transpose_observation(obs, self.mapper),limit=self.limit)
        return new_obs, *other

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple:
        obs, *other = super().reset(seed=seed,options=options)
        new_obs = self._observation_limit(self._transpose_observation(obs, self.mapper),limit=self.limit)
        return new_obs, *other

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        obs = self._transpose_observation(
            self.unwrapped.create_observation(self.unwrapped._cluster),
            self.mapper
        )
        return self.unwrapped._renderer(
            obs,
            current_time=self.unwrapped.time,
            error=self.unwrapped._action_error
        )
