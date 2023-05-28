from ClusterEnv import Dailation, Action, ActionType, PrometheusPuller
import tensorflow as tf
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    pooler = PrometheusPuller(
        url='http://127.0.0.1:49509',
        selected_metrics=[
            "cpu",
            "gpu",
            # "memory Usage",
            # "net_bandwidth Usage",
        ],
        desired_shape=(3,1)
    )
    print(pooler.current)
    example_size = tf.random.normal(shape=(1,20, 20, 3),dtype=tf.float64)
    # dailated = Dailation(example_size, (6,6))
    # print(dailated[0,0])
    # foward = Action(ActionType.Foward,(1,1))
    # Backword = Action(ActionType.Backward)
    # dailated.move(foward)
    # print(dailated[0,0])
    # dailated.move(Backword)
    # print(dailated[0,0])



