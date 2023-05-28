from typing import Tuple, List, Union, NewType, Optional
from attrs import define, field
from tensorflow import keras
import tensorflow as tf
import enum, logging


class ActionType(enum.Enum):
    Foward = enum.auto()
    Backward = enum.auto()


@define
class Action:
    _value: Optional[Tuple[int, int]]
    _type: ActionType


@define
class Dailation:
    orig: tf.Tensor  # size of 4
    pool_shape: tf.TensorShape  # size of 2
    _layers: List[tf.Tensor] = field(init=False)

    def __getitem__(self, indecies: Union[Tuple[int,int],int]) -> tf.Tensor:
        match indecies:
            case int(idx):
                pass
            case int(x_idx,y_idx):
                pass
            case _:
                # TODO: add type error message
                raise TypeError("")
        
        
    def __attrs_post_init__(self):
        _shape = self.orig.shape
        num_examples, num_channels = tf.constant([self.orig.shape[0]]), tf.constant(
            [self.orig.shape[-1]]
        )
        desired_shape = tf.concat([num_examples, self.pool_shape, num_channels], axis=0)
        logging.info(f"Desire Show shape is {desired_shape}")
        self.orig: tf.Tensor = self.__pad_to_fit_pool(self.orig, desired_shape)
        logging.info(f"Padded origin from {_shape} into {self.orig.shape}")
        self._layers = self.__build__layers(self.orig, desired_shape)
        logging.info(f"Build {len(self._layers)} Layers")

    @classmethod
    def __pad_to_fit_pool(cls, inp: tf.Tensor, _pool: tf.TensorShape):
        # TODO: make sure the input and pool shape are the same
        n_times_pool_in_input = tf.math.ceil(tf.math.divide(inp.shape, _pool))
        n_times_pool_in_input = tf.cast(n_times_pool_in_input, dtype=tf.int32)
        # return the
        desire_shape = tf.math.multiply(n_times_pool_in_input, _pool)
        padding_needed_one_side = tf.subtract(desire_shape, inp.shape)
        padding_needed = tf.repeat(padding_needed_one_side, repeats=2)
        padding = tf.divide(tf.reshape(padding_needed, (-1, 2)), 2)
        padding = tf.dtypes.cast(tf.math.ceil(padding), tf.int32)
        return tf.pad(inp, padding)

    @classmethod
    def __build__layers(
        cls, orig: tf.Tensor, _shape: tf.TensorShape
    ) -> List[tf.Tensor]:
        _layers = list()
        logging.debug(f"Building Layers got {orig.shape} ,{_shape}")
        pooler_shape = _shape[1:-1]
        shape_in_the_desired_shape = lambda inp, desire: tf.reduce_all(
            tf.less_equal(inp.shape, desire)
        )
        # TODO: check that the original size is of dim(4)
        pooler = keras.layers.AveragePooling2D(
            pool_size=pooler_shape, strides=pooler_shape
        )
        layer = orig
        while not shape_in_the_desired_shape(layer, _shape):
            logging.debug(f"Added To Layers {layer.shape}")
            _layers.append(layer)
            layer = pooler(layer)
        _layers.append(layer)
        logging.debug(f"Added To Layers {layer.shape}")
        return _layers
