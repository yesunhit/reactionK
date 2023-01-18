
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from eli5.permutation_importance import get_score_importances
from math import sqrt


std_x = StandardScaler()
std_y = StandardScaler()


def load_data(file_path):
    descriptor = pd.read_excel(file_path, header=0)
    return descriptor


def set_all_feature_data():
    data_train = load_data('data/split_data_mutual/train_data_weiguan.xlsx')
    # data_valid = load_data('data/split_data_mutual/valid_data.xlsx')
    data_test = load_data('data/split_data_mutual/test_data_weiguan.xlsx')

    x_train = data_train.iloc[:, 1:data_train.shape[1]]
    columns = x_train.columns
    y_train = data_train.iloc[:, 0]
    x_test = data_test.iloc[:, 1:data_test.shape[1]]
    y_test = data_test.iloc[:, 0]

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # y_train = std_y.fit_transform(y_train)
    # y_valid = std_y.fit_transform(y_valid)
    # y_test = std_y.fit_transform(y_test)

    return x_train, y_train, x_test, y_test, columns


def score(x, y, model):
    y_pred = model.predict(x[0:])
    # y_pred = std_y.inverse_transform(y_pred)
    # y = std_y.inverse_transform(y)
    # return getRMSE(y_pred, y)
    return r2_score(y, y_pred)


def permutation_importance1(x_valid, y_valid, model):
    base_score, score_decreases = get_score_importances(score, x_valid, y_valid, n_iter=10, model=model)
    feature_importances = np.mean(score_decreases, axis=0)
    score_decreases = pd.DataFrame(score_decreases)

    return score_decreases
    # result2 = pd.concat([score_decreases], axis=1)
    # result2.columns = ["R2_value", "deta_R2"]
    # writer = pd.ExcelWriter("data/PI_R2/PMFP_R2_1.xlsx")
    # pd.DataFrame(score_decreases).to_excel(writer, sheet_name='sheet1', index=False)
    # writer.save()


def set_filter_new_index_data(new_index):
    x_train, y_train, x_test, y_test, columns = set_all_feature_data()
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    x_train = x_train.iloc[:, new_index]
    x_test = x_test.iloc[:, new_index]
    x_train.columns = columns[new_index]
    x_test.columns = columns[new_index]
    x_train = x_train.values
    x_test = x_test.values
    return x_train, y_train, x_test, y_test, columns


def LR_model(index):
    x_train_data, y_train_data, x_test, y_test, columns = set_all_feature_data()
    filepath_score = "data/result/LR_PI_score.xlsx"
    n_ford = 10
    train_count = 1
    feature_count = 46
    ford_index = 0
    kf = KFold(n_ford, True)
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
            model = LinearRegression()
            model.fit(x_train, y_train)

            score_decrese = permutation_importance1(x_test, y_test, model)
            score_decrese = score_decrese.values[0]

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

            # print("验证集结果:")
            # print(valid_result)
            for k in range(feature_count):
                score_result[k][i] = score_decrese[k]

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

        score_train = r2_score(mean_train_result, y_train)
        score_valid = r2_score(mean_valid_result, y_valid)
        score_test = r2_score(mean_test_result, y_test)

        print("%f\t%f\t%f\t%f" % (score_train, score_valid, score_test, getRMSE(y_test, mean_test_result)))

        print("The R^2 of train is %f" % score_train)
        print("The R^2 of valid is %f" % score_valid)
        print("The R^2 of test is %f" % score_test)
        print("The RMSE of test is %f" % getRMSE(y_test, mean_test_result))
        # print("The RMSE of valid is %f" % getRMSE(y_valid, mean_valid_result))

    for i in range(len(ford_score_result)):
        temp = 0
        for j in range(len(ford_score_result[1])):
            temp += ford_score_result[i][j]
        mean_ford_score_result[i][0] = temp / len(ford_score_result[1])

    columns_result = [[0 for x in range(1)] for y in range(feature_count)]
    for i in range(len(columns)):
        columns_result[i][0] = columns[i]

    save_result(columns_result, np.array(mean_ford_score_result), filepath_score)


def LR_model_PI():
    data = load_data("data/result/LR_PI_score_new.xlsx")
    data = data.values
    filter_index = []
    for i in range(len(data)):
        # filter_index.append(i)
        if data[i][1] > 0.05:
            filter_index.append(i)
    x_train_data, y_train_data, x_test, y_test, columns = set_filter_new_index_data(filter_index)
    y_train_data = y_train_data.values
    y_test = y_test.values
    model = LinearRegression()
    model.fit(x_train_data, y_train_data)

    print("系数")
    print(model.coef_)

    # 5 模型评估
    # 线性核函数 模型评估
    print(y_test)
    linear_svr_y_predict = model.predict(x_test)
    y_train_predict = model.predict(x_train_data)
    linear_svr_y_predict = model.predict(x_test)
    # y_test = std_y.inverse_transform(y_test)
    # linear_svr_y_predict = std_y.inverse_transform(linear_svr_y_predict)
    print(linear_svr_y_predict)

    # save_result(y_test, np.array(linear_svr_y_predict), filepath)

    ResidualSquare = (linear_svr_y_predict - y_test) ** 2  # 计算残差平方
    print(ResidualSquare)
    RSS = sum(ResidualSquare)  # 计算残差平方和
    MSE = np.mean(ResidualSquare)  # 计算均方差
    num_regress = len(linear_svr_y_predict)  # 回归样本个数
    RMSE = MSE ** 0.5
    score = r2_score(linear_svr_y_predict, y_test)
    score_train = r2_score(y_train_data, y_train_predict)
    scatter_plot(y_test, linear_svr_y_predict)  # 生成散点图

    print(f'n={num_regress}')
    print(f'R^2={score}')
    print(f'R^2={score_train}')
    print(f'RMSE={RMSE}')
    print(f'MSE={MSE}')
    print(f'RSS={RSS}')

    ############绘制折线图##########
    plt.figure()
    plt.plot(np.arange(len(linear_svr_y_predict)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(linear_svr_y_predict)), linear_svr_y_predict, 'ro-', label='predict value')
    plt.title('SVRegressionScatterPlot R^2: %f' % score)
    plt.legend()  # 将样例显示出来
    plt.show()

    save_result(y_test, linear_svr_y_predict, "data/result/test_reslut_lr_weiguan.xlsx")
    save_result(y_train_data, y_train_predict, "data/result/train_reslut_lr_weiguan.xlsx")
    return model



def save_result(y_test, y_new, filepath):
    y_test = pd.DataFrame(y_test)
    writer = pd.ExcelWriter(filepath)
    y_test['new'] = y_new
    pd.DataFrame(y_test).to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()


##########3.绘制验证散点图########
def scatter_plot(TureValues,PredictValues):
    #设置参考的1：1虚线参数
    xxx = [-0.5,1.5]
    yyy = [-0.5,1.5]
    #绘图
    plt.figure()
    plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)#绘制虚线
    plt.scatter(TureValues, PredictValues, s=20, c='r', edgecolors='k', marker='o', alpha=0.8)#绘制散点图，横轴是真实值，竖轴是预测值
    plt.xlim((0, 1))   #设置坐标轴范围
    plt.ylim((0, 1))
    plt.title('SVRegressionScatterPlot')
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
    # print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE

    return sqrt(sum(squaredError) / len(squaredError))


# LR_model(0)
LR_model_PI()
