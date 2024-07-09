import logging
import time

from gymnasium.wrappers.record_video import RecordVideo

from envs.cluster.env import ClusterEnv


def main():

    # logging.basicConfig(level=logging.INFO)
    env = ClusterEnv(n_machines=3, n_jobs=10, n_resource=3, max_ticks=10, render_mode="human")
    # env = RecordVideo(env, './video', episode_trigger=lambda episode_id: True)

    env.reset()
    # algorithm = FirstComeFirstServed(n_jobs)
    for i in range(50):
        action = env.action_space.sample()
        observation, reward, *_ = env.step(action)
        # env.render()
        # action = [1, 1, 1]
    env.close()


if __name__ == "__main__":
    main()
    # main(4)
