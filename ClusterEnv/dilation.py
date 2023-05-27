from typing import Tuple, List, Union, NewType, Optional
from attrs import define, field
import tensorflow as tf
import enum

class ActionType(enum.Enum):
    Foward = enum.auto()
    Backward = enum.auto()


@define
class Action:
    _value: Optional[Tuple[int, int]]
    _type: ActionType


@define
class Dailation:
    orig: tf.Tensor
    desired: tf.TensorShape[int, int, int, int]
    





###############

from attrs import define, field
from typing import Tuple, List, Union, NewType
from enum import Enum , auto
from tensorflow import keras
import tensorflow as tf


class DailationAction(Enum):
    Foward = auto,
    Backward = auto,

BackwordOption = NewType('BackwordOption',DailationAction.Backward)
FowardOption = NewType('FowardOption', Tuple[DailationAction.Foward, Tuple[int, int]])

@define
class Dilation:
    _zero_view_tensor: tf.Tensor
    _output_shape: Tuple[int,int,int]
    _current_position: Tuple[int,int,int] = field(init=False)
    _mapper: List[tf.Tensor] = field(init=False)

    def __getitem__(self, indecies: Union[Tuple[int,int],int]) -> tf.Tensor:
        def get_by_x_y(x: int, y: int):
            x, y = x * self._output_shape[0], y * self._output_shape[1]
            x += self._current_position[1]
            y += self._current_position[2]
            self._current_position = [self._current_position[0], x, y]
            end_x = x + self._output_shape[0]
            end_y = y + self._output_shape[1] 
            print(f'[:, {x}:{end_x} , {y}:{end_y} ,:]')
            return self.mapper[self.current_layer][:, x:end_x , y:end_y ,:]
        in_zero_view = self.current_layer == 0
        in_last_view = self.current_layer >= len(self._mapper)
        match indecies:
            case int(idx) if idx == -1:
                idx_x = idx % self._output_shape[0]
                idx_y = idx // self._output_shape[0]
                if not in_last_view:
                    self.current_layer += 1
                _ , x, y = self._current_position[0]
                end_x = x + self._output_shape[0]
                end_y = y + self._output_shape[1] 
                print(f'[:, {x}:{end_x} , {y}:{end_y} ,:]')
                return self.mapper[self.current_layer][:, x:end_x , y:end_y ,:]
            case int(idx):
                idx_x = idx % self._output_shape[0]
                idx_y = idx // self._output_shape[0]
                if not in_zero_view:
                    self.current_layer -= 1
                return get_by_x_y(idx_x, idx_y)
            case int(idx_x),int(idx_y):
                if not in_zero_view:
                    self.current_layer -= 1
                return get_by_x_y(idx_x, idx_y)
            case _:
                raise ValueError("Some Value Error")

    @classmethod
    def _padd_to_fit_pool(cls, x: tf.Tensor, pool: tf.TensorShape):
        assert len(pool) == len(x.shape)
        new_shape_mult = tf.cast(tf.math.ceil(tf.math.divide(x.shape, pool)), dtype=tf.int32)
        new_shape = tf.math.multiply(new_shape_mult, pool)
        # x the size we want 
        padding = tf.subtract(new_shape, x.shape)
        # Repeat each element twice
        padding = tf.repeat(padding, repeats=2)
        # Reshape the tensor
        padding = tf.divide(tf.reshape(padding, (-1, 2))    ,2)
        padding = tf.dtypes.cast(tf.math.ceil(padding), tf.int32)
        return tf.pad(x, padding)

    @classmethod
    def _create_mapper(cls, zero_view: tf.Tensor, target_size: tf.TensorShape) -> List[tf.Tensor]:
        assert len(target_size) == 2
        max_pooler = keras.layers.AveragePooling2D(pool_size=target_size,strides=target_size)
        left, right = tf.constant([zero_view.shape[0]]),  tf.constant([zero_view.shape[-1]])
        target_size = tf.concat([ left, target_size, right], axis=0)
        level_tensor: tf.Tensor = cls._padd_to_fit_pool(zero_view, target_size)
        bigger_than_target = lambda x: not tf.reduce_all(tf.less_equal(x.shape, target_size))
        mapper: list = []
        while bigger_than_target(level_tensor):
            mapper.append(level_tensor)
            level_tensor = max_pooler(level_tensor)
        mapper.append(level_tensor)
        return mapper

    def __attrs_post_init__(self):
        self.zero_view_tensor = self._zero_view_tensor

    @property
    def current_layer(self):
        return self._current_position[0]

    @property
    def zero_view_tensor(self) -> tf.Tensor:
        return self._zero_view_tensor

    @property
    def mapper(self) -> List[tf.Tensor]:
        return self._mapper

    @current_layer.setter
    def current_layer(self, layer_num: int):
        cur , x, y = self._current_position
        going_back = lambda: cur < layer_num
        if going_back:
            x /= self._output_shape[0]
            y /= self._output_shape[0]
        else: 
            x *= self._output_shape[0]
            y *= self._output_shape[0]
        # print(f"Update Current Layer: {[layer_num, x, y]}")
        self._current_position = [layer_num, int(x), int(y)]

    @mapper.setter
    def mapper(self, value: List[tf.Tensor]):
        self._mapper = self._create_mapper(self._zero_view_tensor, self._output_shape[:-1])
        self._current_position = [len(self.mapper)-1, 0, 0]

    @zero_view_tensor.setter
    def zero_view_tensor(self, zero_view: tf.Tensor):
        self._zero_view_tensor = tf.identity(zero_view)
        self.mapper = self._create_mapper(self._zero_view_tensor, self._output_shape[:-1])
