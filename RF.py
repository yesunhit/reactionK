import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import xlrd
import xlwt
import random
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
###########4.预设回归方法##########
####随机森林回归####
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
import pydot

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

    # x_train = std_x.fit_transform(x_train)
    # x_valid = std_x.transform(x_valid)
    # x_test = std_x.transform(x_test)

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
    print(features.sort_values("score", ascending=False))
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


###########2.回归部分##########
def regression_method(index):
    if index == 0:
        x_train, y_train, x_test, y_test = set_all_feature_data()
        filepath = 'result/all_feature_rf/result_gwp_all_feature.xlsx'
    if index == 1:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_pca_data(0.95)
        filepath = 'result/pca95_rf/result_gwp_pca95.xlsx'
    if index == 2:
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_pca_data(0.90)
        filepath = 'result/pca90_rf/result_gwp_pca90.xlsx'

    model = ensemble.RandomForestRegressor(n_estimators=40, oob_score=False,
                                           n_jobs=-1,
                                           min_samples_split=2,
                                           random_state=50, max_depth=15)  # esitimators决策树数量

    # x_train = np.concatenate((x_train, x_valid), axis=0)
    # y_train = np.concatenate((y_train, y_valid), axis=0)

    model.fit(x_train, y_train)

    importances = model.feature_importances_

    print("重要性：", importances)

    indices = np.argsort(importances)[:: -1]

    print(indices)

    threshold = 0.15

    # x_selected = x_train[:, importances > threshold]
    #
    # print(x_selected)

    result_train = model.predict(x_train)
    result = model.predict(x_test)
    ResidualSquare = (result - y_test)**2     #计算残差平方
    RSS = sum(ResidualSquare)   #计算残差平方和
    MSE = np.mean(ResidualSquare)       #计算均方差
    num_regress = len(result)   #回归样本个数
    RMSE = MSE ** 0.5
    ########5.设置参数与执行部分#############
    # 设置数据参数部分
    # score = model.score(result, y_test)
    score_train = r2_score(result_train, y_train)
    score = r2_score(result, y_test)
    scatter_plot(result, y_test)  # 生成散点图
    print(f'n={num_regress}')
    print(f'Train_R^2={score_train}')
    print(f'R^2={score}')
    print(f'RMSE={RMSE}')
    print(f'MSE={MSE}')
    print(f'RSS={RSS}')
############绘制折线图##########
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('RandomForestRegression R^2: %f'%score)
    plt.legend()        # 将样例显示出来
    plt.show()

    save_result(y_train, result_train, 'data/result/train_reslut_rf.xlsx')
    save_result(y_test, result, 'data/result/test_reslut_rf.xlsx')

    return result


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
    plt.title('RandomForestRegressionScatterPlot')
    plt.show()


def save_result(y_test, y_new, filepath):
    y_test = pd.DataFrame(y_test)
    writer = pd.ExcelWriter(filepath)
    y_test['new'] = y_new
    pd.DataFrame(y_test).to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()


regression_method(0)