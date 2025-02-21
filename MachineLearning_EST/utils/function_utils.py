
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from eli5.permutation_importance import get_score_importances
from math import sqrt
from data_processing import DataProcessing


data_processing = DataProcessing("../data/train_data.xlsx", "../data/test_data.xlsx")


def score(x, y, model):
    y_pred = model.predict(x[0:])
    return r2_score(y, y_pred)


def get_RMSE(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    return sqrt(sum(squaredError) / len(squaredError))


def permutation_importance(x_valid, y_valid, model):
    base_score, score_decreases = get_score_importances(score, x_valid, y_valid, n_iter=10, model=model)
    # feature_importances = np.mean(score_decreases, axis=0)
    score_decreases = pd.DataFrame(score_decreases)
    return score_decreases


def save_result(y_test, y_new, filepath):
    y_test = pd.DataFrame(y_test)
    writer = pd.ExcelWriter(filepath)
    y_test['new'] = y_new
    pd.DataFrame(y_test).to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()


def get_pi_data(file_score_path, model):

    x_train_data, y_train_data, x_test, y_test, columns = data_processing.get_all_feature_data()
    y_train_data = y_train_data.values
    y_test = y_test.values
    filepath_score = file_score_path
    n_ford = 10
    train_count = 1
    feature_count = x_train_data.shape[1]
    ford_index = 0
    kf = KFold(n_ford)
    ford_score_result = [[0 for x in range(n_ford)] for y in range(feature_count)]
    mean_ford_score_result = [[0 for x in range(1)] for y in range(feature_count)]

    for train_index, valid_index in kf.split(x_train_data):
        x_train = x_train_data[train_index]
        y_train = y_train_data[train_index]
        x_valid = x_train_data[valid_index]
        y_valid = y_train_data[valid_index]

        test_result = [[0 for x in range(train_count)] for y in range(len(y_test))]
        train_result = [[0 for x in range(train_count)] for y in range(len(y_train))]
        valid_result = [[0 for x in range(train_count)] for y in range(len(y_valid))]

        score_result = [[0 for x in range(train_count)] for y in range(feature_count)]

        for i in range(train_count):
            model.fit(x_train, y_train)
            score_decrese = permutation_importance(x_test, y_test, model)
            score_decrese = score_decrese.values[0]

            y_new_test = model.predict(x_test[0:])
            print(y_new_test)
            y_new_test = y_new_test.tolist()

            for k in range(len(y_test)):
                test_result[k][i] = y_new_test[k]

            y_new_train = model.predict(x_train[0:])
            y_new_train = y_new_train.tolist()
            for k in range(len(y_train)):
                train_result[k][i] = y_new_train[k]

            y_new_valid = model.predict(x_valid[0:])

            y_new_valid = y_new_valid.tolist()
            for k in range(len(y_valid)):
                valid_result[k][i] = y_new_valid[k]

            for k in range(feature_count):
                score_result[k][i] = score_decrese[k]

        mean_train_result = [[0 for x in range(1)] for y in range(len(y_train))]
        mean_valid_result = [[0 for x in range(1)] for y in range(len(y_valid))]
        mean_test_result = [[0 for x in range(1)] for y in range(len(y_test))]
        mean_score_result = [[0 for x in range(1)] for y in range(len(score_result))]

        print(test_result)
        for i in range(len(test_result)):
            temp = 0
            for j in range(len(test_result[0])):
                temp += test_result[i][j]
            mean_test_result[i][0] = temp / len(test_result[0])

        for i in range(len(train_result)):
            temp = 0
            for j in range(len(train_result[0])):
                temp += train_result[i][j]
            mean_train_result[i][0] = temp / len(train_result[0])

        for i in range(len(valid_result)):
            temp = 0
            for j in range(len(valid_result[0])):
                temp += valid_result[i][j]
            mean_valid_result[i][0] = temp / len(valid_result[0])

        for i in range(len(score_result)):
            temp = 0
            for j in range(len(score_result[1])):
                temp += score_result[i][j]
            mean_score_result[i][0] = temp / len(score_result[1])

        for k in range(len(mean_score_result)):
            ford_score_result[k][ford_index] = mean_score_result[k][0]

        ford_index = ford_index + 1

        score_train = r2_score(y_train, mean_train_result)
        score_valid = r2_score(y_valid, mean_valid_result)
        score_test = r2_score(y_test, mean_test_result)

    for i in range(len(ford_score_result)):
        temp = 0
        for j in range(len(ford_score_result[1])):
            temp += ford_score_result[i][j]
        mean_ford_score_result[i][0] = temp / len(ford_score_result[1])

    columns_result = [[0 for x in range(1)] for y in range(feature_count)]
    for i in range(len(columns)):
        columns_result[i][0] = columns[i]

    save_result(columns_result, np.array(mean_ford_score_result), filepath_score)
