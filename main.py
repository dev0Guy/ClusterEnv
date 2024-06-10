from src.cluster.env import ClusterEnvironment
import logging


def main():

    logging.basicConfig(level=logging.INFO)
    cluster = ClusterEnvironment(n_machines=3, n_jobs=10, n_resources=2, time=10)
    cluster.reset()
    # algorithm = FirstComeFirstServed(n_jobs)
    action = [1, 0, 0]
    for i in range(50):
        observation, *_ = cluster.step(action)
        print(observation)
        cluster.render()
        if i == 1:
            break
        action = [1,1,1]


if __name__ == "__main__":
    main()
    # main(4)
