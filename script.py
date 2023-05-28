from ClusterEnv import Dailation
import tensorflow as tf
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    example_size = tf.random.normal(shape=(1,20, 20, 3),dtype=tf.float64)
    dailated = Dailation(example_size, (6,6))
    dailated[0]
