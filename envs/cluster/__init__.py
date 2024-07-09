from envs.cluster.wrapper import ClusterWrapper
from envs.cluster.env import ClusterEnv
import gymnasium as gym

gym.envs.register(
    id="Cluster-v0",
    entry_point="envs.cluster:ClusterEnv",
    max_episode_steps=100,
    kwargs=dict(
        n_machines=2,
        n_jobs=5,
        n_resource=3,
        max_ticks=4,
    ),
)
