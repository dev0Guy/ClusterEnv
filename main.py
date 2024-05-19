import clusterEnv
import gymnasium as gym
import tianshou as ts
import torch.nn as nn
from clusterEnv.algorithem import FirstComeFirstServed, ShortestJobFirst
import torch
import numpy as np


def main():
    import time

    n_jobs: int = 10
    cluster = gym.make(
        "cluster-v0", nodes=3, jobs=n_jobs, resource=2, time=8, render_mode="rgb_array"
    )
    observation, _ = cluster.reset()
    # algorithm = FirstComeFirstServed(n_jobs)
    algorithm = ShortestJobFirst(n_jobs)
    for _ in range(50):
        action = algorithm.select(observation)
        observation, reward, terminate, trunced, _ = cluster.step(action)
        # time.sleep(1)
        if terminate:
            break


if __name__ == "__main__":
    main()
    # main(4)
