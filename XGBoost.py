import webbrowser

import eli5
from eli5.sklearn import PermutationImportance
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot
from sklearn.model_selection import KFold
from eli5.permutation_importance import get_score_importances
from math import sqrt
from sklearn.inspection import permutation_importance

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
    columns = x_train.columns
    y_train = data_train.iloc[:, 0:1]
    # x_valid = data_valid.iloc[:, 1:data_train.shape[1]]
    # y_valid = data_valid.iloc[:, 0:1]
    x_test = data_test.iloc[:, 1:data_train.shape[1]]
    y_test = data_test.iloc[:, 0:1]
    x_train = std_x.fit_transform(x_train)
    # x_valid = std_x.transform(x_valid)
    x_test = std_x.transform(x_test)

    # y_train = std_y.fit_transform(y_train)
    # y_valid = std_y.transform(y_valid)
    # y_test = std_y.transform(y_test)

    # return x_train, y_train, x_valid, y_valid, x_test, y_test
    return x_train, y_train, x_test, y_test, columns


def score(x, y, model):
    y_pred = model.predict(x[0:])
    y_pred = std_y.inverse_transform(y_pred)
    y = std_y.inverse_transform(y)
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


def main():
    x_train_data, y_train_data, x_test, y_test, columns = set_all_feature_data(1)
    filepath_score = "data/result/XGBoost_PI_score.xlsx"
    n_ford = 10
    train_count = 1
    feature_count = 51
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
            if n_ford == 1:
                break
            model = xgb.XGBRegressor(max_depth=7,
                             min_child_weight=1,
                             learning_rate=0.15,
                             n_estimators=1024,
                             subsample=1,
                             reg_alpha=0.1,
                             reg_lambda=1,
                             # importance_type='cover',
                             gamma=0)

            model.fit(x_train, y_train)

            importance = model.get_booster().get_score(importance_type='cover')
            # importance = model.get_score(model, importance_type='weight')

            print(importance)

            print(model.feature_importances_)

            perm = PermutationImportance(model, random_state=1, n_iter=100, cv=10).fit(x_test, y_test)
            html = eli5.show_weights(perm, feature_names=columns.tolist(), top=30, show_feature_values=True)
            with open('/Users/apple/Downloads/feature_importance.htm', 'wb') as f:
                f.write(html.data.encode("UTF-8"))
                url = r'/Users/apple/Downloads/feature_importance.htm'
                webbrowser.open(url, new=2)
            print(html)

            html_prediction = eli5.show_prediction(model, x_train, feature_names=columns.tolist(),
                                show_feature_values=True)



            score_decrese = permutation_importance(x_test, y_test, model)
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

        y_train = std_y.inverse_transform(y_train)
        mean_train_result = std_y.inverse_transform(mean_train_result)
        # save_result(y_train, np.array(mean_train_result), filepath1)

        y_valid = std_y.inverse_transform(y_valid)
        mean_valid_result = std_y.inverse_transform(mean_valid_result)
        # save_result(y_valid, np.array(mean_valid_result), filepath2)

        mean_test_result = std_y.inverse_transform(mean_test_result)
        # save_result(y_test, np.array(mean_test_result), filepath3)

        score_train = r2_score(mean_train_result, y_train)
        score_valid = r2_score(mean_valid_result, y_valid)
        score_test = r2_score(mean_test_result, y_test)

        print("%f\t%f\t%f\t%f" % (score_train, score_valid, score_test, getRMSE(y_test, mean_test_result)))

        print("The R^2 of train is %f" % score_train)
        print("The R^2 of valid is %f" % score_valid)
        print("The R^2 of test is %f" % score_test)
        print("The RMSE of test is %f" % getRMSE(y_test, mean_test_result))
        print("The RMSE of valid is %f" % getRMSE(y_valid, mean_valid_result))

    for i in range(len(ford_score_result)):
        temp = 0
        for j in range(len(ford_score_result[1])):
            temp += ford_score_result[i][j]
        mean_ford_score_result[i][0] = temp / len(ford_score_result[1])

    columns_result = [[0 for x in range(1)] for y in range(feature_count)]
    for i in range(len(columns)):
        columns_result[i][0] = columns[i]

    save_result(columns_result, np.array(mean_ford_score_result), filepath_score)

    # test_predict=model.predict(X_test)
    # train_predict = model.predict(X_train)
    # xgb.plot_importance(model, ax=None, height=0.2, xlim=None, ylim=None, title='Feature importance',
    #                         xlabel='F score', ylabel='Features', fmap='', importance_type='weight',
    #                         max_num_features=None, grid=True, show_values=True)
    # pyplot.show()
    msetest=mean_squared_error(y_test, test_predict)
    msetrain=mean_squared_error(y_train, train_predict)
    print(sqrt(msetest))
    print(msetrain)
    print(y_test)
    print(test_predict)
    print(abs(test_predict-y_test))
    print(r2_score(y_train, train_predict))
    print(r2_score(test_predict, y_test))
    save_result(y_test, test_predict, 'data/result/test_reslut.xlsx')
    save_result(y_train, train_predict, 'data/result/train_reslut.xlsx')
    print("The RMSE of test is %f" % getRMSE(y_test, test_predict))


def without_CV():
    x_train, y_train, x_test, y_test, columns = set_all_feature_data(1)
    filepath_score = "data/result/XGBoost_PI_score.xlsx"

    feature_count = 50
    train_count = 1

    test_result = [[0 for x in range(train_count)] for y in range(len(y_test))]
    train_result = [[0 for x in range(train_count)] for y in range(len(y_train))]
    score_result = [[0 for x in range(train_count)] for y in range(feature_count)]

    model = xgb.XGBRegressor(max_depth=7,
                             min_child_weight=1,
                             learning_rate=0.1,
                             n_estimators=1024,
                             subsample=1,
                             reg_alpha=1,
                             reg_lambda=1,
                             gamma=0)

    model.fit(x_train, y_train)

    # importance = model.get_booster().get_score(importance_type='weight')
    # # importance = model.get_score(model, importance_type='weight')
    #
    # print(importance)
    #
    # plot_importance(model)
    #
    # pyplot.show()

    # PI特征重要度图
    fig = plt.figure(figsize=(10.8, 9))
    fig.set_dpi(300)
    result = permutation_importance(
        model, x_train, y_train, n_repeats=15, random_state=12, n_jobs=1
    )
    sorted_idx = result.importances_mean.argsort()
    print(sorted_idx)
    # weiguan
    # sorted_idx1 = [37, 42, 39, 48, 45, 35, 33, 46, 34, 36, 49,
    #                 41, 31, 32, 38]

    # 宏观
    # sorted_idx1 = [12, 2, 9, 25, 5, 11, 7, 10, 4, 26, 6,
    #                18, 28, 0, 3]

    # total
    sorted_idx1 = [9, 25, 5, 11, 7, 10, 4, 26, 32, 6, 18,
                  28, 0, 38, 3]

    font = {'style': 'normal',
            'weight': 520,
            'family': 'sans-serif',
            'size': 24,
            }  # 设置字体的风格

    df_up = ['E$_\mathrm{HOMO}$', 'BO$_\mathrm{x}$',
             'f(+)$_\mathrm{x}$', 'f(0)$_\mathrm{x}$',
             'BO$_\mathrm{n}$', 'f(+)$_\mathrm{n}$',
             'E$_\mathrm{LUMO}$', 'q(H$^\mathrm{+}$)',
             'q(C$^\mathrm{-})$$_\mathrm{x}$', 'f(-)$_\mathrm{n}$',
             'E$_\mathrm{B3LYP}$', 'μ',
             'q(CH$^\mathrm{+})$$_\mathrm{n}$', 'q(CH$^\mathrm{+})$$_\mathrm{x}$',
             'E$_\mathrm{gap}$',
             ]

    df_up_total = ['CD',
             'E$_\mathrm{gap}$', 'c(pollutant)',
             'E(NaClO$_\mathrm{4}$)', 'C(Cu)',
             'c(mix)', 'q(CH$^\mathrm{+}$)$_\mathrm{x}$',
             'NaCl', 'V',
             'BDD', 'c(electrolyte)',
             'TiSO', 'electrode',
             'NaBr',
             'pH'
             ]

    rev_list = df_up_total[::-1]


    plt.xticks(fontsize=28, fontproperties=font)
    plt.yticks(fontsize=28, fontproperties=font)
    plt.tick_params(direction='out', length=6, width=2, colors='black',
                 )
    bwith = 3  # 边框宽度设置为2
    TK = plt.gca()  # 获取边框
    TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    TK.spines['left'].set_linewidth(bwith)  # 图框左边
    TK.spines['top'].set_linewidth(bwith)  # 图框上边
    TK.spines['right'].set_linewidth(bwith)  # 图框右边

    print(result.importances)

    x_major_locator = MultipleLocator(0.05)  # 以每15显示
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.boxplot(
        result.importances[sorted_idx1].T,
        vert=False,
        patch_artist=True,
        widths=0.8,
        medianprops={'color': 'black', 'linewidth': '1.8'},
        boxprops={'color': 'black', 'facecolor': '#17b3b7', 'linewidth': '1.5'},
        whiskerprops={'linewidth': '2'},
        flierprops={"marker": "o", "markerfacecolor": "black", "markersize": 6},
        capprops={"linewidth": 2},
        # labels=np.array(columns)[sorted_idx1],
        labels=rev_list,
    )

    plt.title("")
    plt.grid(True)
    plt.grid(linewidth=1.5, alpha=0.2)
    fig.tight_layout()
    plt.show()

    #PI特征重要度结束

   # perm = PermutationImportance(model, random_state=1, n_iter=10, cv=10).fit(x_train, y_train)
    # html = eli5.show_weights(perm, feature_names=columns.tolist(), top=30, show_feature_values=True)
    # with open('/Users/apple/Downloads/feature_importance.htm', 'wb') as f:
    #     f.write(html.data.encode("UTF-8"))
    #     url = r'/Users/apple/Downloads/feature_importance.htm'
    #     webbrowser.open(url, new=2)
    # print(html)

    # html_prediction = eli5.show_prediction(model, x_train, feature_names=columns.tolist(),
    #                     show_feature_values=True)

    # score_decrese = permutation_importance(x_train, y_train, model)
    # score_decrese = score_decrese.values[0]
    #
    # print(score_decrese)

    y_new_test = model.predict(x_test[0:])

    # y_new_test = std_y.inverse_transform(y_new_test)

    print(y_new_test)

    for k in range(len(y_test)):
        test_result[k][0] = y_new_test[k]

    y_new_train = model.predict(x_train[0:])
    y_new_train = y_new_train.tolist()
    for k in range(len(y_train)):
        train_result[k][0] = y_new_train[k]
    # print("验证集结果:")
    # print(valid_result)
    # for k in range(feature_count):
    #     score_result[k][0] = score_decrese[k]

    # y_test = std_y.inverse_transform(y_test)

    score_train = r2_score(y_new_train, y_train)
    score_test = r2_score(y_new_test, y_test)

    print(y_test)

    # print("%f\t%f\t%f\t%f" % (score_train, score_test, score_test, getRMSE(y_test, y_new_test)))

    print("The R^2 of train is %f" % score_train)
    print("The R^2 of test is %f" % score_test)
    print("The RMSE of test is %f" % getRMSE(y_test.values, y_new_test))
    # save_result(y_train, np.array(y_new_train),
    #              "data/result/train_reslut_xbg_weiguan.xlsx")
    # save_result(y_test, np.array(y_new_test), "data/result/test_reslut_xgb_weiguan.xlsx")


def save_result(y_test, y_new, filepath):
    y_test = pd.DataFrame(y_test)
    writer = pd.ExcelWriter(filepath)
    y_test['new'] = y_new
    pd.DataFrame(y_test).to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()


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


# main()
without_CV()