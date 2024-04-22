import gymnasium as gym
import clusterenv
import logging


def main():
    logging.basicConfig(level=logging.INFO)
    env = gym.make('cluster-v0', render_mode='')
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
    pass

if __name__ == "__main__":
    main()