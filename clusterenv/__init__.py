from gymnasium.envs.registration import register

register(
    id='cluster-v0',
    entry_point='clusterenv.envs.cluster:ClusterEnv',
    kwargs=dict(
        nodes=5,
        jobs=10,
        resource=3,
        max_time=5,
        cooldown=1.0,
    )
)
