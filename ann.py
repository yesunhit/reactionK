import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from tensorflow.keras import layers
from keras.layers import LeakyReLU
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from keras import backend as K
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
from eli5.permutation_importance import get_score_importances
from keras.models import load_model
from sklearn.model_selection import KFold

std_x = StandardScaler()
std_y = StandardScaler()


def load_data(file_path):
    descriptor = pd.read_excel(file_path, header=0)
    return descriptor


def set_all_feature_data(category):
    data_train = load_data('data/split_data_mutual/train_data.xlsx')
    # data_valid = load_data('data/split_data_mutual/valid_data.xlsx')
    data_test = load_data('data/split_data_mutual/test_data.xlsx')
    x_train = data_train.iloc[:, 1:data_train.shape[1]]
    y_train = data_train.iloc[:, 0:1]
    # x_valid = data_valid.iloc[:, 1:data_train.shape[1]]
    # y_valid = data_valid.iloc[:, 0:1]
    x_test = data_test.iloc[:, 1:data_train.shape[1]]
    y_test = data_test.iloc[:, 0:1]
    x_train = std_x.fit_transform(x_train)
    # x_valid = std_x.transform(x_valid)
    x_test = std_x.transform(x_test)

    y_train = std_y.fit_transform(y_train)
    # y_valid = std_y.transform(y_valid)
    y_test = std_y.transform(y_test)

    # return x_train, y_train, x_valid, y_valid, x_test, y_test
    return x_train, y_train, x_test, y_test


def set_pca_data(pca, category):
    x_train, y_train, x_test, y_test = set_all_feature_data(category)
    pca = PCA(pca)
    x_train = K.cast_to_floatx(x_train)
    pca.fit(x_train)
    print(pca.n_components_)
    x_train = pca.transform(x_train)
    # x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)

    y_train = K.cast_to_floatx(y_train)
    # y_valid = K.cast_to_floatx(y_valid)
    y_test = K.cast_to_floatx(y_test)
    return x_train, y_train, x_test, y_test


def set_filter_data(num, category):
    data = pd.read_excel('../data/processing.xlsx', header=0, sheet_name='sheet1')
    x_data = data.iloc[:, 1:data.shape[1]]
    y_data = data.iloc[:, 0]
    selector = SelectKBest(score_func=f_regression, k=10)
    # 拟合数据，同时提供特征和目标变量
    results = selector.fit(x_data, y_data)

    print(results)
    # 查看每个特征的分数和p-value
    # results.scores_: 每个特征的分数
    # results.pvalues_: 每个特征的p-value
    # results.get_support(): 返回一个布尔向量，True表示选择该特征，False表示放弃，也可以返回索引
    features = pd.DataFrame({
        "feature": x_data.columns,
        "score": results.scores_,
        "pvalue": results.pvalues_,
        "select": results.get_support()
    })
    features = features.sort_values("score", ascending=True)
    print(features)
    X_new_index = results.get_support(indices=True)
    print(X_new_index)

    x_train, y_train, x_valid, y_valid, x_test, y_test = set_all_feature_data(category)

    x_train = pd.DataFrame(x_train)
    x_valid = pd.DataFrame(x_valid)
    x_test = pd.DataFrame(x_test)
    x_train = x_train.iloc[:, X_new_index]
    x_valid = x_valid.iloc[:, X_new_index]
    x_test = x_test.iloc[:, X_new_index]

    return x_train.values, y_train, x_valid.values, y_valid, x_test.values, y_test


# model1 = load_model('model_MLP_GWP.h5')


def score(x, y, model):
    y_pred = model.predict(x[0:])
    y_pred = std_y.inverse_transform(y_pred)
    y = std_y.inverse_transform(y)
    return r2_score(y, y_pred)


def permutation_importance(x_valid, y_valid, model):
    print(x_valid)
    base_score, score_decreases = get_score_importances(score, x_valid, y_valid, n_iter=1, model=model)
    feature_importances = np.mean(score_decreases, axis=0)
    score_decreases = pd.DataFrame(score_decreases)

    return score_decreases
    # result2 = pd.concat([score_decreases], axis=1)
    # result2.columns = ["R2_value", "deta_R2"]
    # writer = pd.ExcelWriter("data/PI_R2/PMFP_R2_1.xlsx")
    # pd.DataFrame(score_decreases).to_excel(writer, sheet_name='sheet1', index=False)
    # writer.save()


def ann_model(index, num, category, node_count, layer_count, delete_index, new_index):
    if index == 0:
        x_train_data, y_train_data, x_test, y_test = set_all_feature_data(category)
        filepath1 = 'data/result/all_feature_train.xlsx'
        filepath2 = 'data/result/all_feature_valid.xlsx'
        filepath3 = 'data/result/all_feature_test.xlsx'
    if index == 1:
        x_train_data, y_train_data, x_test, y_test = set_pca_data(0.8, category)
        filepath1 = '../data/result/pca95_train.xlsx'
        filepath2 = '../data/result/pca95_valid.xlsx'
        filepath3 = '../data/result/pca95_test.xlsx'

    n_ford = 10
    train_count = 1
    feature_count = 19
    ford_index = 0
    kf = KFold(n_ford, True)
    ford_score_result = [[0 for x in range(n_ford)] for y in range(feature_count)]
    mean_ford_score_result = [[0 for x in range(1)] for y in range(feature_count)]

    y_test = std_y.inverse_transform(y_test)

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
            model = Sequential()
            if layer_count < 2:
                model.add(layers.Dense(node_count, input_shape=(x_train.shape[1],), kernel_initializer='he_uniform'))
                model.add(LeakyReLU(alpha=0.01))
            if layer_count < 3:
                model.add(layers.Dense(node_count, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
                model.add(LeakyReLU(alpha=0.01))
            if layer_count < 4:
                model.add(layers.Dense(node_count, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
                model.add(LeakyReLU(alpha=0.01))
            if layer_count < 5:
                model.add(layers.Dense(node_count, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
                model.add(LeakyReLU(alpha=0.01))
            model.add(layers.Dense(1, activation='linear'))

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01,
                                                           patience=100, verbose=1, mode='auto',
                                                           baseline=None, restore_best_weights=False)

            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['acc'])

            history = model.fit(
                x_train, y_train,
                batch_size=16,
                epochs=2000,
                validation_data=(x_valid, y_valid),  # 验证集
                verbose=0,
                callbacks=[early_stopping])

            # model.save('model_MLP_GWP.h5')
            score_decrese = permutation_importance(x_valid, y_valid, model)
            score_decrese = score_decrese.values[0]

            # draw_fig(history)
            y_new_test = model.predict(x_test[0:])

            y_new_test = y_new_test.tolist()

            for k in range(len(y_test)):
                test_result[k][i] = y_new_test[k][0]

            y_new_train = model.predict(x_train[0:])
            y_new_train = y_new_train.tolist()
            for k in range(len(y_train)):
                train_result[k][i] = y_new_train[k][0]

            y_new_valid = model.predict(x_valid[0:])
            # print("模型在验证集上的结果:")
            # print(y_new_valid)
            # print("模型验证集:")
            # print(y_valid)
            y_new_valid = y_new_valid.tolist()
            for k in range(len(y_valid)):
                valid_result[k][i] = y_new_valid[k][0]

            # print("验证集结果:")
            # print(valid_result)
            # for k in range(feature_count):
            #     score_result[k][i] = score_decrese[k]

        mean_train_result = [[0 for x in range(1)] for y in range(len(y_train))]
        mean_valid_result = [[0 for x in range(1)] for y in range(len(y_valid))]
        mean_test_result = [[0 for x in range(1)] for y in range(len(y_test))]
        # mean_score_result = [[0 for x in range(1)] for y in range(len(score_result))]

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
            # mean_score_result[i][0] = temp / len(score_result[1])

        # for k in range(len(mean_score_result)):
        #     ford_score_result[k][ford_index] = mean_score_result[k][0]

        ford_index = ford_index + 1

        y_train = std_y.inverse_transform(y_train)
        mean_train_result = std_y.inverse_transform(mean_train_result)
        save_result(y_train, np.array(mean_train_result), "data/result/train_reslut_ann_weiguan.xlsx")

        y_valid = std_y.inverse_transform(y_valid)
        mean_valid_result = std_y.inverse_transform(mean_valid_result)
        save_result(y_valid, np.array(mean_valid_result), filepath2)

        mean_test_result = std_y.inverse_transform(mean_test_result)
        save_result(y_test, np.array(mean_test_result), "data/result/test_reslut_ann_weiguan.xlsx")

        score_train = r2_score(mean_train_result, y_train)
        score_valid = r2_score(mean_valid_result, y_valid)
        score_test = r2_score(mean_test_result, y_test)

        print("%f\t%f\t%f\t%f" % (score_train, score_valid, score_test, getRMSE(y_test, mean_test_result)))

        print("The num is %d, The R^2 of train is %f" % (num, score_train))
        print("The num is %d, The R^2 of valid is %f" % (num, score_valid))
        print("The num is %d, The R^2 of test is %f" % (num, score_test))
        print("The num is %d, The RMSE of test is %f" % (num, getRMSE(y_test, mean_test_result)))
        print("The num is %d, The RMSE of valid is %f" % (num, getRMSE(y_valid, mean_valid_result)))

    for i in range(len(ford_score_result)):
        temp = 0
        for j in range(len(ford_score_result[1])):
            temp += ford_score_result[i][j]
        mean_ford_score_result[i][0] = temp / len(ford_score_result[1])
    save_result(mean_ford_score_result, np.array(mean_ford_score_result), filepath1)
    # plt.figure()
    # plt.plot(np.arange(len(mean_test_result)), y_test, 'go-', label='true value')
    # plt.plot(np.arange(len(mean_test_result)), mean_test_result, 'ro-', label='predict value')
    # plt.title('RandomForestRegression R^2: %f' % score_test)
    # plt.legend()  # 将样例显示出来
    # plt.show()

    # return score_valid


def draw_fig(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()


def getRMSE(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    # print("Square Error: ", squaredError)
    # print("Absolute Value of Error: ", absError)
    #
    # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE

    from math import sqrt
    # print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE

    return sqrt(sum(squaredError) / len(squaredError))


def save_result(y_test, y_new, filepath):
    y_test = pd.DataFrame(y_test)
    writer = pd.ExcelWriter(filepath)
    y_test['new'] = y_new
    pd.DataFrame(y_test).to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()


ann_model(0, 234, 1, 256, 1, 0, 0)
# set_filter_data(0, 0)

