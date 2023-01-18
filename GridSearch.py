import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

std_x = StandardScaler()
std_y = StandardScaler()


def load_data(file_path):
    descriptor = pd.read_excel(file_path, header=0)
    return descriptor


def set_all_feature_data():
    data_train = load_data('data/split_data_mutual/train_data.xlsx')
    data_test = load_data('data/split_data_mutual/test_data.xlsx')
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
                         'gamma': [0.01, 0.005, 0.001, 0.0005, 0.0001],
                         'C': [1, 10, 100, 1000]}]
    model = SVR()
    clf = GridSearchCV(model, tuned_parameters, cv=5,
                       scoring='r2')
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()

    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf.best_params_)

    print()
    print("Grid scores on development set:")
    print()


def XGB_grid():
    parameters = {
        'max_depth': [3],
        'learning_rate': [0.15],
        'n_estimators': [200],
        'min_child_weight': [2],
        'max_delta_step': [1],
        'subsample': [1],
        'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    }

    X_train, y_train, X_test, y_test, columns = set_all_feature_data()
    model = xgb.XGBRegressor(max_depth=7,
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


def ann_grid():
    X_train, y_train, X_test, y_test, columns = set_all_feature_data()

# XGB_grid()
SVR_grid()