from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
from typing import List, Dict, Tuple
from attrs import define, field, Factory
import tensorflow as tf
import pandas as pd
import numpy as np


@define
class PrometheusPuller:
    url: str
    selected_metrics: List[str]
    desired_shape: tf.TensorShape  # [metric,pox_x, pos_y]
    _client: PrometheusConnect = field(init=False)
    _metric_mapper: Dict[str, int] = field(init=False)
    _job_mapper: Dict[str, int] = Factory(dict)

    def __attrs_post_init__(self):
        self._client = PrometheusConnect(url=self.url)
        self._metric_mapper = dict(
            zip(self.selected_metrics, range(len(self.selected_metrics)))
        )
        channels = tf.TensorShape((len(self.selected_metrics),))
        self.desired_shape = tf.TensorShape(self.desired_shape)
        self.desired_shape = self.desired_shape.concatenate(channels)

    def __add_new_jobs(self, df: pd.DataFrame):
        not_registered_ndes = filter(
            lambda x: x not in self._job_mapper, df["exported_job"]
        )
        for job in not_registered_ndes:
            self._job_mapper[job] = len(self._job_mapper)

    @classmethod
    def __create_tensor_idx(
        cls,
        df: pd.DataFrame,
        _shape: tf.TensorShape,
        metric_mapper: dict,
        job_mapper: dict,
    ) -> pd.Series:
        name_indexed = df["__name__"].map(metric_mapper)
        job_indexed = df["exported_job"].map(job_mapper)
        job_indexed = job_indexed.apply(lambda x: (x // _shape[1], x % _shape[1])).T
        combined = map(lambda x: (*x[0], x[1]), zip(job_indexed, name_indexed))
        serices = pd.Series(list(combined))
        serices.index = df.index
        return serices

    @classmethod
    def __convert_to_tensor(
        cls,
        df: pd.DataFrame,
        _shape: tf.TensorShape,
        metric_mapper: dict,
        job_mapper: dict,
    ) -> tf.Tensor:
        df["tensor_index"] = cls.__create_tensor_idx(
            df, _shape, metric_mapper, job_mapper
        )
        global_view: np.ndarray = np.zeros(_shape, dtype=np.float64)
        for _, row in df.iterrows():
            global_view[row["tensor_index"]] = row["value"]
        return tf.convert_to_tensor(global_view)

    @property
    def current(self):
        def extract(metric_name):
            metric_df = self._client.get_metric_range_data(metric_name)
            return MetricRangeDataFrame(metric_df).drop_duplicates(
                subset=["__name__", "exported_job"], keep="last"
            )

        cluster_df = pd.concat(iter(map(extract, self.selected_metrics)))
        self.__add_new_jobs(cluster_df)
        return self.__convert_to_tensor(
            cluster_df, self.desired_shape, self._metric_mapper, self._job_mapper
        )
