from envs.cluster import ClusterEnv
from dataclasses import dataclass, field
import numpy.typing as npt
import gymnasium as gym
import numpy as np
@dataclass
class QueueWrapper(gym.Wrapper):
    env: ClusterEnv
    limit: int = field(default=3)

    def __post_init__(self):
        super(QueueWrapper, self).__init__(self.env)
        orig_obs_space: gym.spaces.Dict = self.env.observation_space
        orig_obs_space["Queue"] = gym.spaces.Box(
            low=-1,
            high=np.inf,
            shape=self.obs_shape_size,
            dtype=np.float64,
        )
        self.observation_space = gym.spaces.Discrete(self.limit * self.env.nodes)
        self.action_space = gym.spaces.Discrete(self.limit * self.env.nodes)


# from clusterenv.envs._cluster import ClusterEnv
# from typing import Any, SupportsFloat
# import gymnasium as gym
# import numpy as np

# class QueueLimitWrapper(gym.Wrapper):

#     def __init__(self, env: ClusterEnv, queue_limit: int = 3):
#         super(QueueLimitWrapper, self).__init__(env)
#         self.queue_limit: int = queue_limit
#         self.action_space = gym.spaces.Discrete((self.env.n_nodes * self.queue_limit)+1)
#         self.obs_shape_size: tuple = (self.queue_limit,* self.env.observation_space["Queue"].shape[1:])
#         self.observation_space["Queue"] = gym.spaces.Box(
#             low=-1,
#             high=np.inf,
#             shape=self.obs_shape_size,
#             dtype=np.float64,
#         )
#         self.mapper = np.full(self.env.n_jobs, -1)

#     def transpose_observation(self, obs):
#         queue = np.where(np.all(obs["Queue"] >= 0, axis=(1,2)))[0]
#         new_obs_queue: np.arange = np.full(self.obs_shape_size, -1)
#         self.mapper = queue[:self.queue_limit]
#         new_obs_queue[:len(self.mapper)] = obs["Queue"][self.mapper]
#         obs["Queue"]= new_obs_queue
#         return obs

#     def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
#         if action > 0:
#             n_idx , j_idx = self.env.convert(n_nodes=self.env.n_nodes,action=action-1)
#             action = 0
#             if j_idx < len(self.mapper):
#                 orig_j_idx = self.mapper[j_idx]
#                 action = (orig_j_idx * self.env.n_nodes) + n_idx + 1
#         obs, reward, terminated, trunced, self.info = super().step(action)
#         return self.transpose_observation(obs), reward, terminated, trunced, self.info

#     def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
#         obs, self.info = super().reset(seed=seed,options=options)
#         return self.transpose_observation(obs), self.info

#     def render(self) -> None:
#         obs = self.transpose_observation(self.unwrapped._observation)
#         return self.render_obs(obs)
