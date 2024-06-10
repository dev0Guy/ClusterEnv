from gymnasium.core import ObsType, WrapperObsType
from clusterEnv import Status
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
        return np.concatenate((observation["Nodes"], observation["Jobs"]), axis=0)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        nodes: gym.spaces.Box = self.observation_space["Nodes"]
        jobs: gym.spaces.Box = self.observation_space["Jobs"]
        num_c: int = nodes.shape[0] + jobs.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(num_c, *jobs.shape[1:])
        )
