from gymnasium.core import ObsType, WrapperObsType
from clusterenv._types import JobStatus
import gymnasium as gym
import numpy as np


class ConcatenateObservationDict(gym.ObservationWrapper):
    """
        Concatenate based the following fields:
            - FreeSpace: distance between Nodes, Usage.
            - Queue: uses Queue field and map each status into the queue field.
                     'JobStatus.RUNNING' => 0
                     'obStatus.COMPLETE' => -np.inf
                     'JobStatus.NONEXISTENT' => -1
        Then pad the smaller state (first channel) and concatenate and reshape into
        -> (2, max(jobs, nodes), resource, time)
    """

    def observation(self, observation: ObsType) -> WrapperObsType:
        queue: np.ndarray = observation['Queue']
        status: np.array = observation['Status']
        free_space: np.ndarray = observation['Nodes'] - observation['Usage']
        max_nj = max(free_space.shape[0], queue.shape[0])
        # mapp Jobs
        queue[status == JobStatus.RUNNING] = 0
        queue[status == JobStatus.COMPLETE] = -np.inf
        queue[status == JobStatus.NONEXISTENT] = -1
        #
        free_space = np.pad(free_space, ((0, max_nj - free_space.shape[0]), (0, 0), (0, 0)), mode='constant')
        queue = np.pad(queue, ((0, max_nj - queue.shape[0]), (0, 0), (0, 0)), mode='constant')
        n_shape = (2, max_nj, queue.shape[1], queue.shape[2])
        c_obs: np.ndarray = np.concatenate((free_space, queue), axis=0)\
                            .reshape(n_shape)
        return c_obs

    def _get_observation_space(self) -> gym.spaces.Box:
        max_val: float = np.max(self.observation_space['Nodes'].high)
        n, r, t = self.observation_space['Nodes'].shape
        j, *_ = self.observation_space['Queue'].shape
        n_shape = (2, max(n, j), r, t)
        low: np.ndarray = np.full(n_shape, -np.inf)
        high: np.ndarray = np.full(n_shape, max_val)
        return gym.spaces.Box(low=low, high=high, dtype=float)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self._get_observation_space()
        print(self.observation_space)
