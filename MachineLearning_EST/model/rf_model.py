
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from MachineLearning_EST.utils.data_processing import DataProcessing

data_processing = DataProcessing("../data/train_data.xlsx", "../data/test_data.xlsx")


def PI_RF():
    data = data_processing.load_data("../result/RF_PI_score.xlsx")
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
    kf = KFold(n_ford)
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
            model = set_model()

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

        score_train = r2_score(mean_train_result, y_train)
        score_valid = r2_score(mean_valid_result, y_valid)
        score_test = r2_score(mean_test_result, y_test)
        print(score_test)

    model = set_model()
    model.fit(x_train_data, y_train_data)
    y_test_predicted = model.predict(x_test)
    print(r2_score(y_test, y_test_predicted))


def set_model():
    model = ensemble.RandomForestRegressor(n_estimators=32, oob_score=False,
                                           n_jobs=-1,
                                           min_samples_split=2,
                                           random_state=50,
                                           max_depth=15)
    return model


# function_utils.get_pi_data("../result/RF_PI_score.xlsx", set_model())
PI_RF()
