from gymnasium.envs.registration import register

register(
    id="cluster-v0",
    entry_point="clusterenv.envs.cluster:ClusterEnv",
    kwargs=dict(
        n_nodes=5,
        n_jobs=10,
        n_resource=3,
        max_time=5,
    ),
)
