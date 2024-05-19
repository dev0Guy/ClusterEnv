from .cluster import ClusterEnv
from .cluster import Status
from gymnasium.envs.registration import register

register(
    id="cluster-v0",
    entry_point="clusterEnv.envs.cluster:ClusterEnv",
    kwargs=dict(nodes=2, jobs=10, resource=1, time=1, max_episode_steps=400),
)
