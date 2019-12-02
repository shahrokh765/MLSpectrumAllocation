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

var_f = open("variables_" + datetime.datetime.now().strftime('_%Y%m_%d%H_%M')+ ".txt", "wb")


dataframe = pd.read_csv('ML\data\dynamic_pus_using_pus50000_15PUs_201910_2616_25.txt', delimiter=',', header=None)
dataframe_max = pd.read_csv('ML\data\dynamic_pus_max_power50000_15PUs_201910_2616_25.txt', delimiter=',', header=None)
dataframe.reset_index(drop=True, inplace=True)
dataframe_max.reset_index(drop=True, inplace=True)
dataframe_n = pd.concat([dataframe.iloc[:, 0:len(dataframe.columns)-2], dataframe_max.iloc[:, dataframe_max.columns.values[-1]]], axis=1,
                        ignore_index=True)
idx = dataframe_n[dataframe_n[dataframe_n.columns[-1]] == -float('inf')].index
dataframe_n.drop(idx, inplace=True)
data = dataframe_n.values

number_samples = [5] + list(range(10, 101, 10)) + [120, 150, 200, 250, 300, 400, 500, 700] + list(range(1000, 4001, 1000))
# number_samples = [3000]
validation_size = 0.33   # of training samples
fp_penalty_coef = 100
fn_penalty_coef = 1
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

for samples in number_samples:
    sample = math.ceil(samples * (1 + validation_size))
    X_train = data[0:sample, 0: num_inputs]
    y_train = data[0:sample, -1]
    X_test = data[sample:, 0: num_inputs]
    y_test = data[sample:, -1]

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