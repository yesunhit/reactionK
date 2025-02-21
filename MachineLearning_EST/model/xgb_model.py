
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from MachineLearning_EST.utils.data_processing import DataProcessing

data_processing = DataProcessing("../data/train_data.xlsx", "../data/test_data.xlsx")


def PI_XGBoost():
    data = data_processing.load_data("../result/XGBoost_PI_score.xlsx")
    data = data.values
    filter_index = []
    for i in range(len(data)):
        if data[i][1] > 0:
            filter_index.append(i)
    x_train_data, y_train_data, x_test, y_test, columns = data_processing.get_filter_new_index_data(filter_index)
    y_train_data = y_train_data.values
    y_test = y_test.values
    n_ford = 10
    train_count = 1
    feature_count = x_train_data.shape[1]
    ford_index = 0
    kf = KFold(n_ford)
    ford_score_result = [[0 for x in range(n_ford)] for y in range(feature_count)]
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
            model = set_xgb_model()
            model.fit(x_train, y_train)

            y_new_test = model.predict(x_test[0:])
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

        mean_train_result = [[0 for x in range(1)] for y in range(len(y_train))]
        mean_valid_result = [[0 for x in range(1)] for y in range(len(y_valid))]
        mean_test_result = [[0 for x in range(1)] for y in range(len(y_test))]
        mean_score_result = [[0 for x in range(1)] for y in range(len(score_result))]

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

        #y_train = std_y.inverse_transform(y_train)
        #mean_train_result = std_y.inverse_transform(mean_train_result)
        # save_result(y_train, np.array(mean_train_result), filepath1)

        #y_valid = std_y.inverse_transform(y_valid)
        #mean_valid_result = std_y.inverse_transform(mean_valid_result)
        # save_result(y_valid, np.array(mean_valid_result), filepath2)

        #mean_test_result = std_y.inverse_transform(mean_test_result)
        # save_result(y_test, np.array(mean_test_result), filepath3)

        score_train = r2_score(y_train, mean_train_result)
        score_valid = r2_score(y_valid, mean_valid_result)
        score_test = r2_score(y_test, mean_test_result)
        print(score_train)
        print(score_test)

    model = set_xgb_model()

    model.fit(x_train_data, y_train_data)

    y_test_predicted = model.predict(x_test)
    print(r2_score(y_test, y_test_predicted))


def set_xgb_model():
    model = xgb.XGBRegressor(max_depth=4,
                             min_child_weight=1,
                             learning_rate=0.15,
                             n_estimators=512,
                             subsample=1,
                             reg_alpha=0.1,
                             reg_lambda=1,
                             # importance_type='cover',
                             gamma=0)

    return model


# function_utils.get_pi_data("../result/XGBoost_PI_score.xlsx", set_xgb_model())
PI_XGBoost()
