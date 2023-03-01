import numpy as np
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from keras import models
from keras import layers
from sklearn import ensemble
from keras.wrappers import scikit_learn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
from keras.layers import LeakyReLU
import keras

std_x = StandardScaler()
std_y = StandardScaler()


def load_data(file_path):
    descriptor = pd.read_excel(file_path, header=0)
    return descriptor


def set_all_feature_data():
    data_train = load_data('../data/train_data.xlsx')
    data_test = load_data('../data/test_data.xlsx')
    x_train = data_train.iloc[:, 1:data_train.shape[1]]
    columns = x_train.columns
    y_train = data_train.iloc[:, 0:1]
    x_test = data_test.iloc[:, 1:data_train.shape[1]]
    y_test = data_test.iloc[:, 0:1]
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    return x_train, y_train, x_test, y_test, columns


def SVR_grid():
    X_train, y_train, X_test, y_test, columns = set_all_feature_data()
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.005, 0.001, 0.0005, 0.0001],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5],
                         'gamma': [0.1, 0.01, 0.001, 0.0001],
                         'C': [1, 10, 100, 1000]}]
    model = SVR()
    clf = GridSearchCV(model, tuned_parameters, cv=5,
                       scoring='r2')
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    print()
    print("Grid scores on development set:")
    print()


def XGB_grid():
    parameters = {
        'max_depth': [2, 4, 6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'n_estimators': [64, 128, 256, 512, 1024],
        'min_child_weight': [1, 2, 3, 4, 5],
        'Gamma': [0, 0.25, 0.5, 0.75, 1],
        'reg_alpha': [0.2, 0.4, 0.6, 0.8, 1],
        'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    }

    X_train, y_train, X_test, y_test, columns = set_all_feature_data()
    model = xgb.XGBRegressor(max_depth=2,
                             min_child_weight=1,
                             learning_rate=0.1,
                             n_estimators=1024,
                             subsample=1,
                             reg_alpha=1,
                             reg_lambda=1,
                             gamma=0)

    gsearch = GridSearchCV(model, param_grid=parameters, scoring='r2', cv=3)
    gsearch.fit(X_train, y_train)
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def RF_grid():
    parameters = {
        'max_depth': [5, 10, 15, 20],
        'n_estimators': [32, 64, 128, 256, 512],
        'min_samples_split': [1, 2, 3, 4, 5],
    }

    X_train, y_train, X_test, y_test, columns = set_all_feature_data()
    model = ensemble.RandomForestRegressor(n_estimators=40, oob_score=False,
                                           n_jobs=-1,
                                           min_samples_split=2,
                                           random_state=50,
                                           max_depth=15)

    gsearch = GridSearchCV(model, param_grid=parameters, scoring='r2', cv=3)
    gsearch.fit(X_train, y_train)
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def create_model(activation='relu', init_mode='random_uniform',
                 learning_rate=0.01, node_count=4, layer_count=1):
    # create model
    x_train, y_train = set_all_feature_data()
    model = models.Sequential()
    if layer_count < 2:
        model.add(layers.Dense(node_count, input_shape=(x_train.shape[1],), kernel_initializer=init_mode))
        model.add(LeakyReLU(alpha=0.01))
    if layer_count < 3:
        model.add(layers.Dense(node_count, kernel_regularizer=tf.keras.regularizers.l2(0.005)))
        model.add(LeakyReLU(alpha=0.01))
    if layer_count < 4:
        model.add(layers.Dense(node_count, kernel_regularizer=tf.keras.regularizers.l2(0.005)))
        model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dense(1, activation=activation))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse', metrics=['acc'])
    # model.compile(optimizer=optimizer, loss='rmse', metrics=['accuracy'])
    return model


def batch_epoch():
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_all_feature_data()
    means = np.mean(x_train, axis=0)
    x_train -= means
    stds = np.std(y_train, axis=0)
    x_train /= stds
    seed = 7
    np.random.seed(seed)
    model = scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)
    batch_size = [32, 64, 128, 256]
    epochs = [10, 50, 100, 200, 500]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)
    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))


def ref_function():
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_all_feature_data()
    means = np.mean(x_train, axis=0)
    x_train -= means
    stds = np.std(x_train, axis=0)
    x_train /= stds
    seed = 7
    np.random.seed(seed)
    model = scikit_learn.KerasRegressor(build_fn=create_model, epochs=20, batch_size=256, verbose=0)
    optimizer = ['sgd', 'lbfgs’', 'adam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)
    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))


def learn_rate():
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_all_feature_data()
    means = np.mean(x_train, axis=0)
    x_train -= means
    stds = np.std(x_train, axis=0)
    x_train /= stds
    seed = 7
    np.random.seed(seed)
    model = scikit_learn.KerasRegressor(build_fn=create_model, epochs=20, batch_size=256, verbose=0)
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    param_grid = dict(learning_rate=learning_rate, momentum=momentum)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)
    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))


def init_weight():
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_all_feature_data()
    means = np.mean(x_train, axis=0)
    x_train -= means
    stds = np.std(x_train, axis=0)
    x_train /= stds
    seed = 7
    np.random.seed(seed)
    model = scikit_learn.KerasRegressor(build_fn=create_model, epochs=20, batch_size=256, verbose=0)
    init_mode = ['he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)
    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))


def init_activation():
    x_train, y_train = set_all_feature_data()
    means = np.mean(x_train, axis=0)
    x_train -= means
    stds = np.std(x_train, axis=0)
    x_train /= stds
    seed = 7
    np.random.seed(seed)
    model = scikit_learn.KerasRegressor(build_fn=create_model, epochs=20, batch_size=256, verbose=0)
    activation = ['relu', 'tanh', 'softmax']
    learning_rate = [0.0001, 0.001, 0.01]
    init_mode = ['he_normal', 'he_uniform']
    # optimizer = ['sgd', 'rmsprop', 'adam', 'adagrad']
    # batch_size = [8, 16, 32, 64, 128]
    node_count = [16, 32, 64, 128, 256, 512]
    layer_count = [1, 2, 3]
    param_grid = dict(activation=activation, learning_rate=learning_rate, init_mode=init_mode,
                      node_count=node_count, layer_count=layer_count)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
                                                   patience=50, verbose=1, mode='auto',
                                                   baseline=None, restore_best_weights=False)
    grid_result = grid.fit(x_train, y_train, epochs=2000,
                           batch_size=32,
                           callbacks=[early_stopping])

    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))


# init_activation()
XGB_grid()
SVR_grid()