from gymnasium import ActionWrapper
from gymnasium.core import WrapperActType, ActType
import gymnasium as gym
from src.cluster.env import ClusterEnvironment


class ClusterActionWrapper(ActionWrapper):

    def __init__(self, env: ClusterEnvironment):
        super().__init__(env)
        self.n_machines, self.n_jobs, _ = env.action_space.nvec;

        self.action_space = gym.spaces.Discrete(self.n_machines * self.n_jobs + 1, start=0)

    def action(self, action: int) -> ActType:
        skip_time = 1 if action == 0 else 0
        action -= 1
        m_idx, j_idx = action % self.n_machines, action // self.n_machines
        action = (abs(m_idx), abs(j_idx), int(skip_time))
        return action