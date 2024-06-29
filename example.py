from src.cluster.env import ClusterEnvironment
from gymnasium.wrappers.record_video import RecordVideo
import logging


def main():

    # logging.basicConfig(level=logging.INFO)
    env = ClusterEnvironment(
        n_machines=3, n_jobs=10, n_resources=3, time=10,
        render_mode='human'
    )
    # env = RecordVideo(env, './video', episode_trigger=lambda episode_id: True)

    env.reset()
    # algorithm = FirstComeFirstServed(n_jobs)
    for i in range(50):
        action = env.action_space.sample()
        observation, reward, *_ = env.step(action)
        print(reward)
        # cluster.render()
        # action = [1, 1, 1]
    env.close()


if __name__ == "__main__":
    main()
    # main(4)
