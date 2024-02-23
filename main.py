from clusterenv.envs.cluster import ClusterEnv


if __name__ == "__main__":
    env = ClusterEnv()
    env.render()
    print(env.observation_space)
    print(env.action_space)
    print(env.reset())
    print(env)
