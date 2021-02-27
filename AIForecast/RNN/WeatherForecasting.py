import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.python.keras.callbacks import ModelCheckpoint, History
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential, load_model

from AIForecast import utils
from AIForecast.utils import DataUtils, PathUtils


class TimestepBatchGenerator:
    """
    This class is adapted from https://www.tensorflow.org/tutorials/structured_data/time_series.
    Formats the passed data to be used with a neural network as a future predictive model.

    Takes the data passed and splits the data into batches of timed windows.
    """

    def __init__(
            self,
            train_set: pd.DataFrame, validate_set: pd.DataFrame, test_set: pd.DataFrame,
            data_window, label_window, future_window,
            feature_labels=None
    ):
        """
        train_set, validate_set, test_set - the sets used for time windowing.
        data_window - the length of the time window in which training is sampled from. Indicative of X training set.
        label_window - the length of the window indicative of actual results. Corresponds to the length of data used as
        the y set.
        future_window - the length of the time window in which predictions will be made off of. I.e. "Predict x hours
        into the future". Synonymous with time offset.
        feature_labels - a list of labels used for the actual results used to compare to the predicted results.

        The label window is offset from the data window to create a rolling window for predictions.
        """
        self.train_set, self.validate_set, self.test_set = train_set, validate_set, test_set
        self.feature_labels = feature_labels
        if feature_labels is not None:
            self.feature_indices = {name: i for i, name in enumerate(feature_labels)}
        self.column_indices = {name: i for i, name in enumerate(self.train_set.columns)}
        self.data_window, self.label_window, self.offset = data_window, label_window, future_window
        self.window_size = data_window + future_window
        self.slice_data_window = slice(0, data_window)
        self.data_indices = np.arange(self.window_size)[self.slice_data_window]
        self.label_window_begin = self.window_size - self.label_window
        self.slice_label_window = slice(self.label_window_begin, None)
        self.label_indices = np.arange(self.window_size)[self.slice_label_window]

    def split_window(self, features):
        data = features[:, self.slice_data_window, :]
        labels = features[:, self.slice_label_window, :]
        if self.feature_labels is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.feature_labels], axis=-1)
        data.set_shape([None, self.data_window, None])
        labels.set_shape([None, self.label_window, None])
        return data, labels

    def make_dataset(self, data, stride=1, batch_size=32):
        data = np.array(data, dtype=np.float32)
        data_set = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.window_size,
            sequence_stride=stride,
            shuffle=True,
            batch_size=batch_size
        )
        data_set = data_set.map(self.split_window)
        return data_set

    @property
    def train(self):
        return self.make_dataset(self.train_set)

    @property
    def validate(self):
        return self.make_dataset(self.validate_set)

    @property
    def test(self):
        return self.make_dataset(self.test_set)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.window_size}',
            f'Input indices: {self.data_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.feature_labels}'])


class ForecastingNetwork:

    _MAX_EPOCHS = 50
    """
    The number of times the neural network is fed back.
    """

    def __init__(self, data, batch_size=32):
        self.train, self.validate, self.test = DataUtils.split_data(data)
        self.train_mean, self.train_std = self.train.mean(), self.train.std()
        self.train = self.scale(self.train, self.train_mean, self.train_std)
        self.validate = self.scale(self.validate, self.train_mean, self.train_std)
        self.test = self.scale(self.test, self.train_mean, self.train_std)
        self.model = Sequential([
            LSTM(batch_size, return_sequences=True),
            Dense(units=1)
        ])
        self.history: History = None
        self.generator = None

    def train_network(self, hours_into_the_future, features=None):
        if features is None:
            features = ['temperature']

        batch_generator = TimestepBatchGenerator(
            self.train,
            self.validate,
            self.test,
            int(hours_into_the_future),
            len(features),
            int(hours_into_the_future),
            features
        )
        self.generator = batch_generator
        self.history = self._compile_and_fit(batch_generator)
        utils.log(__name__).debug(self.model.summary())

    def get_example_predictions(self):
        return [self.unscale(pred, self.train_mean['temperature'], self.train_std['temperature'])
                for pred in np.array(self.model.predict(self.generator.example[0])).flatten()]

    def _compile_and_fit(self, generator: TimestepBatchGenerator):
        checkpoint = ModelCheckpoint(
            filepath=PathUtils.get_file(PathUtils.get_model_path(), 'model-{epoch:02d}.hdf5'),
            verbose=1
        )
        self.model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()]
        )
        self._save_mean_std()
        return self.model.fit(
            generator.train,
            epochs=ForecastingNetwork._MAX_EPOCHS,
            validation_data=generator.validate,
            callbacks=[checkpoint]
        )

    def _save_mean_std(self):
        mean_std = {'mean': self.train_mean.to_dict(), 'std': self.train_std.to_dict()}
        with open(PathUtils.get_file(PathUtils.get_model_path(), 'mean_std.json'), 'w') as f:
            json.dump(mean_std, f)

    @staticmethod
    def scale(df: pd.DataFrame, mean: pd.Series, std: pd.Series):
        return (df - mean) / std

    @staticmethod
    def unscale(prediction, mean, std):
        return prediction * std + mean

    @staticmethod
    def get_saved_model():
        """
        Returns a saved predictive model along with a Series containing the mean for each feature and another
        Series containing the standard deviation of each feature.
        """
        model = load_model(PathUtils.get_file(PathUtils.get_model_path(), 'model-50.hdf5'))
        with open(PathUtils.get_file(PathUtils.get_model_path(), 'mean_std.json'), 'r') as f:
            mean_std = json.load(f)
        model_mean = pd.Series(mean_std['mean'])
        model_std = pd.Series(mean_std['std'])
        return model, model_mean, model_std
