import pandas as pd

from keras.models import Sequential
import keras.backend as K
import keras
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import datetime
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt

# number_samples = [5] + list(range(10, 101, 10)) + [120, 150, 200, 250, 300, 400, 500, 700] + list(range(1000, 4001, 1000))
number_samples = [256, 512, 1028, 2048, 4096, 8192]
# number_samples = [3000]
validation_size = 0.33   # of training samples
max_pus_num, max_sus_num = 20, 1
IS_SENSORS, sensors_num = False, 225
DUMMY_VALUE = -90.0
fp_penalty_coef = 1
fn_penalty_coef = 1

num_columns = (sensors_num if IS_SENSORS else max_pus_num * 3 + 1) + max_pus_num * 3 + 2
cols = [i for i in range(num_columns)]
dataframe = pd.read_csv("../../../java_workspace/research/spectrum_allocation/resources/data/" +
                        "dynamic_pus_using_pus50000_15PUs_201910_2616_25.txt",
                        delimiter=',', header=None, names=cols)
dataframe_max = pd.read_csv("../../../java_workspace/research/spectrum_allocation/resources/data/" +
                            "dynamic_pus_max_power50000_15PUs_201910_2616_25.txt",
                            delimiter=',', header=None)
dataframe.reset_index(drop=True, inplace=True)
dataframe_max.reset_index(drop=True, inplace=True)
dataframe_n = pd.concat([dataframe.iloc[:, 0:len(dataframe.columns)-2], dataframe_max.iloc[:, dataframe_max.columns.values[-1]]], axis=1,
                        ignore_index=True)
idx = dataframe_n[dataframe_n[dataframe_n.columns[-1]] == -float('inf')].index
dataframe_n.drop(idx, inplace=True)
data = dataframe_n.values

var_f = open("variables_" + datetime.datetime.now().strftime('_%Y%m_%d%H_%M')+ ".txt", "wb")
average_diff_power = []
fp_mean, fp_count = [], []
max_diff_power = []
num_inputs = data.shape[1] - 1
# samples = 100
# TODO: Apply REGULARIZATION
def custom_loss(fp_penalty_coef, fn_penalty_coef):
    def loss(y_true, y_pred):
        return K.mean(fp_penalty_coef * K.square(y_pred - y_true)[y_pred > y_true]) + \
               K.mean(fn_penalty_coef * K.square(y_pred - y_true)[y_pred <= y_true])
    return loss


def split(data : np.ndarray, train_samples: int, max_pus_number: int, max_sus_number: int, IS_SENSORS: bool,
          num_sensors: int, DUMMY_VALUE: float):
    num_inputs = (max_sus_number - 1) * 3 + 2 + (max_pus_number * 3
                                                 if not IS_SENSORS else num_sensors)
    # val_samples = round(train_samples / 3)
    test_samples = data.shape[0] - train_samples
    # init arrays
    X_train = np.ones((train_samples, num_inputs), dtype=float) * DUMMY_VALUE
    # X_val = np.ones((val_samples, num_inputs), dtype=float) * DUMMY_VALUE
    X_test = np.ones((test_samples, num_inputs), dtype=float) * DUMMY_VALUE
    # read values
    if not IS_SENSORS:
        # fill train
        for train_sample in range(train_samples):
            num_pus = int(data[train_sample, 0])
            num_sus = int(data[train_sample, 1 + num_pus * 3])
            X_train[train_sample, :num_pus * 3] = data[train_sample, 1:1 + num_pus * 3]  # pus
            # sus except power of last su
            X_train[train_sample, max_pus_number * 3:
                                  (max_pus_number + num_sus) * 3 - 1] = \
                data[train_sample, 2 + num_pus * 3: 1 + (num_pus + num_sus) * 3]
        # fill test
        for test_sample in range(train_samples, train_samples + test_samples):
            num_pus = int(data[test_sample, 0])
            num_sus = int(data[test_sample, 1 + num_pus * 3])
            X_test[test_sample - train_samples, :num_pus * 3] = data[test_sample, 1:1 + num_pus * 3]
            X_test[test_sample - train_samples, max_pus_number * 3:
                                                (max_pus_number + num_sus) * 3 - 1] = \
                data[test_sample, 2 + num_pus * 3:1 + (num_pus + num_sus) * 3]
    else:
        # read sensors
        X_train[:, :num_sensors] = data[:train_samples, :num_sensors]
        X_test[:, :num_sensors] = data[train_samples:, :num_sensors]
        # read sus
        for train_sample in range(train_samples):
            num_sus = int(data[train_sample, num_sensors])
            X_train[train_sample, num_sensors + 1: num_sensors + num_sus * 3] = data[
                train_sample, num_sensors + 1:num_sensors + num_sus * 3]

        for test_sample in range(train_samples, train_samples + test_samples):
            num_sus = int(data[test_sample, num_sensors])
            X_test[test_sample - train_samples, num_sensors + 1: num_sensors + num_sus * 3] =\
                data[test_sample, num_sensors + 1:num_sensors + num_sus * 3]

    y_train = data[0: train_samples, -1]
    # y_val = data[train_samples: train_samples + val_samples, -1]
    y_test = data[train_samples:, -1]
    return X_train, X_test, y_train, y_test


for samples in number_samples:
    sample = math.ceil(samples * (1 + validation_size))
    # X_train = data[0:sample, 0: num_inputs]
    # y_train = data[0:sample, -1]
    # X_test = data[sample:, 0: num_inputs]
    # y_test = data[sample:, -1]
    X_train, X_test, y_train, y_test = split(data, sample, max_pus_num, max_sus_num,
                                             IS_SENSORS, sensors_num,
                                             DUMMY_VALUE)

    y_train = np.reshape(y_train, (-1,1))
    scaler_x = StandardScaler()
    # scaler_y = StandardScaler()
    scaler_x.fit(X_train)
    X_scale = scaler_x.transform(X_train)
    # scaler_y.fit(y_train)
    # y_scale = scaler_y.transform(y_train)
    # y_scale = y_train

    # X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, shuffle=False, train_size=0.75, test_size=0.25)

    model = Sequential()
    model.add(Dense(20, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, activation='relu'))
    # model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss=custom_loss(fp_penalty_coef, fn_penalty_coef), optimizer='adam', metrics=['mse','mae'])  # loss = {'mse', 'custom_loss'}
    history = model.fit(X_scale, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=validation_size)

    # print(history.history.keys())
    # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    # Xnew = np.array([[40, 0, 26, 9000, 8000]])
    X_test = scaler_x.transform(X_test)
    yp_test = model.predict(X_test)
    #invert normalize
    # yp_test = scaler_y.inverse_transform(yp_test)
    # yp_test = np.reshape(yp_test, (-1, 1))
    y_test_l, yp_test_l = y_test.tolist(), yp_test.tolist()
    sum_, max_, count = 0, -float('inf'), 0
    fp_sum_, fp_cnt = 0, 0
    for i in range(len(y_test_l)):
        count += 1
        diff_temp = abs(y_test_l[i] - yp_test_l[i][0])
        sum_ += diff_temp
        max_ = max(max_, diff_temp)
        if yp_test_l[i][0] > y_test_l[i]:
            fp_sum_ += diff_temp
            fp_cnt += 1

    average_diff_power.append(round(sum_/count, 3))
    fp_mean.append(round(fp_sum_/count, 3))
    fp_count.append(fp_cnt)
    max_diff_power.append(round(np.amax(max_), 3))

pickle.dump([average_diff_power, max_diff_power, fp_mean, number_samples], file=var_f)
    # Xnew = scaler_x.inverse_transform(Xnew)
    # print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(20, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#     model.add(Dense(20, kernel_initializer='normal', activation='relu'))
#     # model.add(Dense(20, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal', activation='linear'))
#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
# # evaluate model
#
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))