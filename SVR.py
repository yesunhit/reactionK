from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from keras import backend as K
import matplotlib.pyplot as plt

std_x = StandardScaler()
std_y = StandardScaler()


def load_data(file_path):
    descriptor = pd.read_excel(file_path, header=0)
    return descriptor


def set_all_feature_data():
    data_train = load_data('data/split_data_mutual/train_data.xlsx')
    # data_valid = load_data('data/split_data_mutual/valid_data.xlsx')
    data_test = load_data('data/split_data_mutual/test_data.xlsx')
    x_train = data_train.iloc[:, 1:data_train.shape[1]]
    y_train = data_train.iloc[:, 0]
    # x_valid = data_valid.iloc[:, 1:data_train.shape[1]]
    # y_valid = data_valid.iloc[:, 0]
    x_test = data_test.iloc[:, 1:data_train.shape[1]]
    y_test = data_test.iloc[:, 0]

    x_train = std_x.fit_transform(x_train)
    # x_valid = std_x.transform(x_valid)
    x_test = std_x.transform(x_test)

    # y_train = std_y.fit_transform(y_train)
    # y_valid = std_y.fit_transform(y_valid)
    # y_test = std_y.fit_transform(y_test)

    return x_train, y_train, x_test, y_test


def set_pca_data(pca):
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_all_feature_data()
    pca = PCA(pca)
    x_train = K.cast_to_floatx(x_train)
    pca.fit(x_train)
    print(pca.n_components_)
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)

    y_train = K.cast_to_floatx(y_train)
    y_valid = K.cast_to_floatx(y_valid)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def set_filter_data():
    data = pd.read_excel('data/feature.xlsx', header=0, sheet_name='sheet1')
    x_data = data.iloc[:, 0:data.shape[1] - 4]
    y_data = data.iloc[:, data.shape[1] - 4]

    selector = SelectKBest(score_func=f_regression, k=400)
    # 拟合数据，同时提供特征和目标变量
    results = selector.fit(x_data, y_data)
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
    features.sort_values("score", ascending=False)
    X_new_index = results.get_support(indices=True)

    x_train, y_train, x_valid, y_valid, x_test, y_test = set_all_feature_data()

    x_train = pd.DataFrame(x_train)
    x_valid = pd.DataFrame(x_valid)
    x_test = pd.DataFrame(x_test)
    x_train = x_train.iloc[:, X_new_index]
    x_valid = x_valid.iloc[:, X_new_index]
    x_test = x_test.iloc[:, X_new_index]

    return x_train.values, y_train, x_valid.values, y_valid, x_test.values, y_test


def set_filter_pca_data():
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_filter_data()
    pca = PCA(0.98)
    x_train = K.cast_to_floatx(x_train)
    pca.fit(x_train)
    print(pca.n_components_)
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)

    y_train = K.cast_to_floatx(y_train)
    y_valid = K.cast_to_floatx(y_valid)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def set_wrapper_data():
    data_train = load_data('data/wrapper_data/wrapper_train.xlsx')
    data_valid = load_data('data/wrapper_data/wrapper_valid.xlsx')
    data_test = load_data('data/wrapper_data/wrapper_test.xlsx')
    x_train = data_train.iloc[:, 0:data_train.shape[1] - 4]
    y_train = data_train.iloc[:, data_train.shape[1] - 4]
    x_valid = data_valid.iloc[:, 0:data_train.shape[1] - 4]
    y_valid = data_valid.iloc[:, data_train.shape[1] - 4]
    x_test = data_test.iloc[:, 0:data_train.shape[1] - 4]
    y_test = data_test.iloc[:, data_train.shape[1] - 4]

    x_train = std_x.fit_transform(x_train)
    x_valid = std_x.transform(x_valid)
    x_test = std_x.transform(x_test)

    # y_train = std_y.fit_transform(y_train)
    # y_valid = std_y.fit_transform(y_valid)
    # y_test = std_y.fit_transform(y_test)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def set_wrapper_pca_data():
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_wrapper_data()
    pca = PCA(0.95)
    x_train = K.cast_to_floatx(x_train)
    pca.fit(x_train)
    print(pca.n_components_)
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)

    y_train = K.cast_to_floatx(y_train)
    y_valid = K.cast_to_floatx(y_valid)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def set_embedded_data():
    data_train = load_data('data/embedded_data/embedded_train.xlsx')
    data_valid = load_data('data/embedded_data/embedded_valid.xlsx')
    data_test = load_data('data/embedded_data/embedded_test.xlsx')
    x_train = data_train.iloc[:, 0:data_train.shape[1] - 4]
    y_train = data_train.iloc[:, data_train.shape[1] - 4]
    x_valid = data_valid.iloc[:, 0:data_train.shape[1] - 4]
    y_valid = data_valid.iloc[:, data_train.shape[1] - 4]
    x_test = data_test.iloc[:, 0:data_train.shape[1] - 4]
    y_test = data_test.iloc[:, data_train.shape[1] - 4]

    x_train = std_x.fit_transform(x_train)
    x_valid = std_x.transform(x_valid)
    x_test = std_x.transform(x_test)

    # y_train = std_y.fit_transform(y_train)
    # y_valid = std_y.fit_transform(y_valid)
    # y_test = std_y.fit_transform(y_test)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def set_embedded_pca_data():
    x_train, y_train, x_valid, y_valid, x_test, y_test = set_embedded_data()
    pca = PCA(0.95)
    x_train = K.cast_to_floatx(x_train)
    pca.fit(x_train)
    print(pca.n_components_)
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)

    y_train = K.cast_to_floatx(y_train)
    y_valid = K.cast_to_floatx(y_valid)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def SVR_model(index):
    if index == 0:
        x_train, y_train, x_test, y_test = set_all_feature_data()
        filepath = 'result/all_feature_svr/result_gwp_all_feature_1.xlsx'
    if index == 1:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_pca_data(0.95)
        filepath = 'result/pca95_svr/result_gwp_pca95_1.xlsx'
    if index == 2:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_pca_data(0.90)
        filepath = 'result/pca90_svr/result_gwp_pca90_1.xlsx'
    if index == 3:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_pca_data(0.85)
        filepath = 'result/pca85_svr/result_gwp_pca85_1.xlsx'
    if index == 4:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_filter_data()
        filepath = 'result/filter_ann/result_filter.xlsx'
    if index == 5:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_filter_pca_data()
        filepath = 'result/filter_ann/result_filter_pca95.xlsx'
    if index == 6:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_wrapper_data()
        filepath = 'result/wrapper_ann/result_wrapper.xlsx'
    if index == 7:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_wrapper_pca_data()
        filepath = 'result/wrapper_ann/result_wrapper_pca95.xlsx'
    if index == 8:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_embedded_data()
        filepath = 'result/embedded_ann/result_embedded.xlsx'
    if index == 9:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_embedded_pca_data()
        filepath = 'result/embedded_ann/result_embedded_pca95.xlsx'

    linear_svr = SVR(kernel='poly', degree=5, tol=0.0001, gamma=0.01, coef0=1, C=300)
    linear_svr.fit(x_train, y_train)
    linear_svr_y_predict = linear_svr.predict(x_test)
    linear_svr_y_predict_train = linear_svr.predict(x_train)
    poly_svr = SVR(kernel="poly")
    poly_svr.fit(x_train, y_train)
    poly_svr_y_predict = linear_svr.predict(x_test)
    # 5 模型评估
    # 线性核函数 模型评估
    print(y_test)
    # y_test = std_y.inverse_transform(y_test)
    # linear_svr_y_predict = std_y.inverse_transform(linear_svr_y_predict)
    print(linear_svr_y_predict)

    # save_result(y_test, np.array(linear_svr_y_predict), filepath)

    ResidualSquare = (linear_svr_y_predict - y_test) ** 2  # 计算残差平方
    RSS = sum(ResidualSquare)  # 计算残差平方和
    MSE = np.mean(ResidualSquare)  # 计算均方差
    num_regress = len(linear_svr_y_predict)  # 回归样本个数
    RMSE = MSE ** 0.5
    score = r2_score(linear_svr_y_predict, y_test)
    score_train = r2_score(linear_svr_y_predict_train, y_train)
    scatter_plot(y_test, linear_svr_y_predict)  # 生成散点图

    print(f'n={num_regress}')
    print(f'R^2={score_train}')
    print(f'R^2={score}')
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

    # mean_result = [[0 for x in range(1)] for y in range(len(y_test))]
    # for i in range(len(test_result)):
    #     temp = 0
    #     for j in range(len(test_result[0])):
    #         temp += test_result[i][j]
    #         print(temp)
    #     mean_result[i][0] = temp / len(test_result[0])
    #
    # print(test_result)
    # print(np.array(mean_result))

    save_result(y_train, linear_svr_y_predict_train, "data/result/train_reslut_svr_weiguan.xlsx")
    save_result(y_test, linear_svr_y_predict, "data/result/test_reslut_svr_weiguan.xlsx")
    # return model, history


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


SVR_model(0)