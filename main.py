from src.cluster.env import ClusterEnvironment
import logging

def main():

    logging.basicConfig(level=logging.INFO)
    cluster = ClusterEnvironment(n_machines=3, n_jobs=10, n_resources=2, time=10)
    cluster.reset()
    # algorithm = FirstComeFirstServed(n_jobs)
    for _ in range(50):
        action = 0
        _ = cluster.step(action)
        cluster.render()


if __name__ == "__main__":
    main()
    # main(4)
