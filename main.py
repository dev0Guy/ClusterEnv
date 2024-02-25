import logging
logging.basicConfig(level=logging.INFO)
from clusterenv.envs.cluster import ClusterEnv
if __name__ == "__main__":
    env = ClusterEnv(nodes=3,jobs=11)
    # env.reset()
    for idx in range(100):
        env.render()
        action: int =  0 if idx % 5 == 0 else env.action_space.sample()
        env.step(action)
