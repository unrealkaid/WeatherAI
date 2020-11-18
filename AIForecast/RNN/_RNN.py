import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from AIForecast.utils import PathUtils

data_path = '/'
offset_in_hours = 8


# counts data size of pkl files
def counting_pkl_size():
    path_utils = PathUtils()
    train_path = path_utils.get_pkl_path() + path_utils.TRAINING_FILE
    valid_path = path_utils.get_pkl_path() + path_utils.VALIDATION_FILE
    test_path = path_utils.get_pkl_path() + path_utils.TESTING_FILE

    train_size = 0
    valid_size = 0
    test_size = 0

    with open(train_path, 'rb') as f1:
        while True:
            try:
                temp = pickle.load(f1)
            except EOFError:
                break
            train_size = train_size + 1
    f1.close()

    with open(valid_path, 'rb') as f2:
        while True:
            try:
                temp = pickle.load(f2)
            except EOFError:
                break
            valid_size = valid_size + 1
    f2.close()

    with open(test_path, 'rb') as f3:
        while True:
            try:
                temp = pickle.load(f3)
            except EOFError:
                break
            test_size = test_size + 1
    f3.close()
    return train_size, valid_size, test_size


# For getting needed y (temperature) values for greater offsets, say 48 hours instead of 1
# Needs the paths to each data file and the desired offset (in hours)
def get_y_data(train_path, valid_path, test_path, hours_offset):
    train_y = []
    valid_y = []
    test_y = []
    with open(train_path, 'rb') as train:
        for x in range(hours_offset):
            temp = pickle.load(train)
        while True:
            try:
                train_y.append(pickle.load(train)[3][4])
            except EOFError:
                break
    with open(valid_path, 'rb') as valid:
        for x in range(hours_offset):
            temp = pickle.load(valid)
        while True:
            try:
                valid_y.append(pickle.load(valid)[3][4])
            except EOFError:
                break
    with open(test_path, 'rb') as test:
        for x in range(hours_offset):
            temp = pickle.load(test)
        while True:
            try:
                test_y.append(pickle.load(test)[3][4])
            except EOFError:
                break
    return train_y, valid_y, test_y


class KerasBatchGenerator(object):

    def __init__(self, data, y, num_steps, batch_size, file_size, skip_step=1):
        self.data = data
        self.y = y
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.file_size = file_size
        self.current_spot = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    # Batch generation function. Creates batches of data equal to batch_size
    # num_steps determines how many hours are used for each piece of data fed in
    # num_steps = 4 means 4 hours will be used in making predictions each time
    # the output will have as many values as num_steps (feed in 4 hours, it will predict 4 hours)
    def generate(self):
        temp_y = []
        while True:
            if self.current_spot > (
                    self.file_size - 3 * (batch_size + (num_steps - 1))):  # reset if not enough data left for a batch
                self.current_spot = 0
            with open(self.data, 'rb') as f:
                x = np.zeros((self.batch_size + (num_steps - 1), 3), dtype=float)
                y2 = np.zeros((self.batch_size, num_steps, 1), dtype=float)
                for t in range(self.current_spot):
                    temp = pickle.load(f)
                for i in range(self.batch_size + (num_steps - 1)):
                    if i == 0:
                        temp_x = pickle.load(f)
                        temp_x = np.delete(temp_x, 3, axis=0)  # removes erie data row
                        x[i] = temp_x[0][3], temp_x[1][3], temp_x[2][3]
                        # assign the temp values of the cities that are not erie
                    else:
                        x[i] = temp_y[0][3], temp_y[1][3], temp_y[2][3]
                    temp_y = pickle.load(f)
                    if i < batch_size:  # makes sure this doesn't go out of range
                        for t in range(num_steps):
                            y2[i][t][0] = self.y[i + self.current_spot + t]
                    temp_y = np.delete(temp_y, 3, axis=0)  # removes erie data row
                    self.current_spot += 1
                # this section of code creates the array of 'windows' used as input. num_steps is the size of window
                x2 = np.zeros((batch_size, num_steps, 3))
                for i in range(batch_size):
                    temp = []
                    for z in range(num_steps):
                        temp.append(x[i + z])
                    x2[i] = temp
                yield x2, y2
            f.close()


num_steps = 1
batch_size = 72


class RNN:
    # this function will run the RNN with whatever data is currently in the .pkl files
    train_data, valid_data, test_data = " ", " ", " "
    train_size, valid_size, test_size = 0, 0, 0
    train_Y, valid_Y, test_Y = [], [], []

    def run_rnn(self):
        path_utils = PathUtils()
        self.train_data = path_utils.get_pkl_path() + path_utils.TRAINING_FILE
        self.valid_data = path_utils.get_pkl_path() + path_utils.VALIDATION_FILE
        self.test_data = path_utils.get_pkl_path() + path_utils.TESTING_FILE

        self.train_size, self.valid_size, self.test_size = counting_pkl_size()

        self.train_Y, self.valid_Y, self.test_Y = get_y_data(self.train_data, self.valid_data, self.test_data,
                                                             offset_in_hours)

        train_data_generator = KerasBatchGenerator(self.train_data, self.train_Y, num_steps, batch_size,
                                                   self.train_size,
                                                   skip_step=1)
        valid_data_generator = KerasBatchGenerator(self.valid_data, self.valid_Y, num_steps, batch_size,
                                                   self.valid_size,
                                                   skip_step=1)
        hidden_size = 160
        use_dropout = True
        model = Sequential()
        model.add(LSTM(hidden_size, input_shape=(num_steps, 3), return_sequences=True,
                       kernel_initializer=keras.initializers.VarianceScaling(),
                       recurrent_initializer=keras.initializers.VarianceScaling()))
        model.add(Dropout(.3))
        model.add(LSTM(hidden_size, return_sequences=True, kernel_initializer=keras.initializers.VarianceScaling(),
                       # 'orthogonal'
                       recurrent_initializer=keras.initializers.VarianceScaling(), activation='relu'))
        if use_dropout:
            model.add(Dropout(0.25))
            model.add(Activation('relu'))
        model.add(TimeDistributed(Dense(1)))

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adagrad(lr=.005),
                      metrics=['mean_absolute_percentage_error'])

        print(model.summary())
        check_pointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
        num_epochs = 50
        model.fit_generator(train_data_generator.generate(), self.train_size / (batch_size - 2), num_epochs,
                            validation_data=valid_data_generator.generate(),
                            validation_steps=self.valid_size / (batch_size - 2), callbacks=[check_pointer])

        model = load_model(data_path + "/model-50.hdf5")
        dummy_iterators = 50
        example_training_generator = KerasBatchGenerator(self.valid_data, self.valid_Y, num_steps, batch_size,
                                                         self.valid_size,
                                                         skip_step=1)
        return_string = ""
        return_string = return_string + "Training is complete.\n Here are some testing results: \n\n"
        return_string = return_string + "Validation data:\n"
        for i in range(dummy_iterators):
            dummy = next(example_training_generator.generate())
        num_predict = 50
        accuracy = 0
        for i in range(num_predict):
            data, y = next(example_training_generator.generate())
            test = data[i]
            test = np.reshape(test, (1, num_steps, 3))
            prediction = model.predict(test)
            actual = ((y[i] * (323.0 - 244.0) + 244.0) - 273.15) * (9.0 / 5.0) + 32.0
            prediction = ((prediction * (323.0 - 244.0) + 244.0) - 273.15) * (9.0 / 5.0) + 32.0
            if abs(actual - prediction) < 10:
                accuracy += 1
            return_string = return_string + "Actual: " + str(actual) + "\n"
            return_string = return_string + "Prediction: " + str(prediction) + "\n"
        accuracy = accuracy / num_predict
        return_string = return_string + "\nAccuracy: " + str(accuracy * 100) + "\n\n"

        # test data set
        dummy_iterators = 10
        example_test_generator = KerasBatchGenerator(self.test_data, self.test_Y, num_steps, batch_size, self.test_size,
                                                     skip_step=1)
        return_string = return_string + "Test data:\n"
        for i in range(dummy_iterators):
            dummy = next(example_test_generator.generate())
        num_predict = 50
        accuracy = 0
        for i in range(num_predict):
            data, y = next(example_test_generator.generate())
            test = data[i]
            test = np.reshape(test, (1, num_steps, 3))
            prediction = model.predict(test)
            actual = ((y[i] * (323.0 - 244.0) + 244.0) - 273.15) * (9.0 / 5.0) + 32.0
            prediction = ((prediction * (323.0 - 244.0) + 244.0) - 273.15) * (9.0 / 5.0) + 32.0
            if abs(actual - prediction) < 10:
                accuracy += 1
            return_string = return_string + "Actual: " + str(actual) + "\n"
            return_string = return_string + "Prediction: " + str(prediction) + "\n"
        accuracy = accuracy / num_predict
        return_string = return_string + "\nAccuracy: " + str(accuracy * 100) + "\n\n"
        print(return_string)
        return return_string
