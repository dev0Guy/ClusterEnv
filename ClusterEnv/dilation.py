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
    _type: ActionType
    _value: Optional[Tuple[int, int]] = field(default=None)


@define
class Dailation:
    orig: tf.Tensor  # size of 4
    pool_shape: tf.TensorShape  # size of 2
    _layers: List[tf.Tensor] = field(init=False)
    _current_layer: int = field(
        init=False, default=0
    )  # TODO: make property and chack its not negative
    _position_in_layer: Tuple[int, int] = field(init=False, default=[0, 0])

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
        self._current_layer = len(self._layers) - 1
        logging.info(f"Build {len(self._layers)} Layers")

    def __select_item_index(
        self, indecies: Union[Tuple[int, int], int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        match indecies:
            case int(idx):
                x_idx = idx % self.pool_shape[0]
                y_idx = idx // self.pool_shape[0]
                return self.__convert_index_to_current_layer(
                    x_idx, y_idx, self.pool_shape, self._position_in_layer
                )
            case int(x_idx, y_idx):
                return self.__convert_index_to_current_layer(
                    x_idx, y_idx, self.pool_shape, self._position_in_layer
                )
            case _:
                # TODO: add type error message
                raise TypeError("")

    def __getitem__(self, indecies: Tuple[int, int]) -> tf.Tensor:
        x  = indecies[0] + self._position_in_layer[0]
        y  = indecies[1] + self._position_in_layer[1]
        logging.debug(f"Getting [:, {x}, {y}, :] from layer {self._current_layer} ")
        return self._layers[self._current_layer][:, x, y, :]

    def __foward(self, indecies: Union[Tuple[int, int], int]):
        x, y = indecies
        org_layer = self._current_layer
        self._position_in_layer[0] = x * self.pool_shape[0]
        self._position_in_layer[1] = y * self.pool_shape[1]
        self._current_layer -= 1
        logging.debug(f"Moving Foward from layer: {org_layer} index: {indecies} to layer: {self._current_layer} index:{self._position_in_layer}")

    def __backward(self):
        org_layer = self._current_layer
        self._position_in_layer[0] = 0 ### x // self.pool_shape[0]
        self._position_in_layer[1] = 0 ### y // self.pool_shape[1]
        self._current_layer += 1
        logging.debug(f"Moving Backward from layer: {org_layer} layer: {self._current_layer}")

    def move(self, action: Action):
        match action._type:
            case ActionType.Foward  if self._current_layer > 0:
                self.__foward(action._value)
            case ActionType.Backward if self._current_layer < len(self._layers):
                self.__backward()

    @classmethod
    def __convert_index_to_current_layer(
        cls, x: int, y: int, _pool, _current
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x = x * _pool[0] + _current[0]
        y = y * _pool[1] + _current[1]
        x_end = x + _pool[0]
        y_end = y + _pool[1]
        return (x, x_end), (y, y_end)

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
