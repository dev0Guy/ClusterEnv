from gymnasium.envs.registration import register
from _tmo import wrappers

register(
    id="cluster-v0",
    entry_point="_tmo.envs.cluster:ClusterEnv",
    kwargs=dict(
        n_nodes=5,
        n_jobs=10,
        n_resource=3,
        max_time=5,
        cooldown=0.0001,
        max_episode_steps=100,
    ),
)
