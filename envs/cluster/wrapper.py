from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, Wrapper
from gymnasium.core import ActType, WrapperActType, ObsType, WrapperObsType


class _ClusterActionWrapper(ActionWrapper):

    def __init__(self, env: gym.Env, *, reorganize: bool = False):
        super().__init__(env)
        self.reorganize = reorganize
        self.n_machines, self.n_jobs, _ = env.action_space.nvec
        self.action_space = gym.spaces.Discrete(self.n_machines * self.n_jobs + 1, start=0)

    def action(self, action: int) -> ActType:
        assert self.action_space.contains(action)
        skip_time: bool = action == 0
        if skip_time:
            return (1, 0, 0)
        action -= 1
        m_idx, j_idx = action // self.n_machines, action % self.n_machines
        action = (int(skip_time), m_idx, j_idx)
        return action


# TODO:
# - Wrapper for actions
# - Wrapper for sorted and Queue
# - Wrapper for Dilation Machine
# - Wrapper for

class ClusterWrapper(Wrapper):

    def __init__(self, env: gym.Env, *, queue_size: int = np.inf):
        super().__init__(env)
        # NOTICE: problem when both queue size and machine size is 1
        self.index = np.arange(env.n_jobs)
        _, self.n_machines, self.n_jobs = env.action_space.nvec
        self.queue_size = min(queue_size, self.n_jobs)
        assert self.queue_size > 1
        self.observation_space = env.observation_space
        self.observation_space["jobsUsage"] = gym.spaces.Box(0.0, 1.0, (self.queue_size, env.n_resource, env.max_ticks), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.n_machines * self.queue_size + 1, start=0)

    def observation(self, observation: ObsType) -> WrapperObsType:
        observation = observation.copy()
        # TODO: given observation return extra information
        status: np.ndarray = observation["jobStatus"]
        self.index: np.ndarray = np.argsort(status)
        observation["jobStatus"] = status[self.index][: self.queue_size]
        observation["jobsUsage"] = observation["jobsUsage"][self.index][: self.queue_size]
        return observation

    def cast_action(self, action: WrapperActType) -> ActType:
        skip_time: bool = action == 0
        if skip_time:
            return (True, 0, 0)
        action -= 1
        m_idx, j_idx = (action % self.n_machines, action // self.n_machines)
        action = (skip_time, m_idx, j_idx)
        return action

    def action(self, action: WrapperActType) -> ActType:
        assert self.action_space.contains(action)
        action = self.cast_action(action)
        skip, m_idx, j_idx = action
        j_idx = self.index[j_idx]
        return (skip, m_idx, j_idx)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = self.action(action)
        obs, *other = self.env.step(action)
        obs = self.observation(obs)
        return obs, *other

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        obs = self.observation(obs)
        return obs, info


class ClusterDilationWrapper(Wrapper):

    def __init__(self, env: gym.Env, *, kernel: int | tuple[int, int] = (3, 3)):
        self.kernel: tuple[int, int] = kernel
        super().__init__(env)
        if isinstance(self.kernel, tuple):
            pass
        elif isinstance(kernel, int):
            self.kernel = (self.kernel, self.kernel)
        else:
            raise ValueError
        self.n_machines: int = np.prod(self.kernel)
        self.action_space = gym.spaces.Discrete(self.n_machines * self.n_jobs + 1, start=0)
        # TODO: extract observation size

