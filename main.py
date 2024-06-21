from src.cluster.env import ClusterEnvironment
import logging
import time

def main():

    logging.basicConfig(level=logging.INFO)
    cluster = ClusterEnvironment(n_machines=3, n_jobs=10, n_resources=3, time=10, render_mode='human')
    cluster.reset()
    # algorithm = FirstComeFirstServed(n_jobs)
    action = [1, 0, 0]
    for i in range(50):
        observation, *_ = cluster.step(action)
        cluster.render()
        # action = [1, 1, 1]


if __name__ == "__main__":
    main()
    # main(4)
