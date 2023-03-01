import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

std_x = StandardScaler()
std_y = StandardScaler()
loaded_model = pickle.load(open("model.dat", "rb"))

result_x = []
result_y = []


def load_data(file_path):
    descriptor = pd.read_excel(file_path, header=0)
    return descriptor


def set_all_feature_data():
    data_train = load_data('data/train_data.xlsx')
    data_test = load_data('data/test_data.xlsx')
    x_train = data_train.iloc[:, 1:data_train.shape[1]]
    columns = x_train.columns
    y_train = data_train.iloc[:, 0:1]
    x_test = data_test.iloc[:, 1:data_train.shape[1]]
    y_test = data_test.iloc[:, 0:1]
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    return x_train, y_train, x_test, y_test, columns


class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high, x_test, x_train):
        self.dimension = dimension
        self.time = time
        self.size = size
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))
        self.v = np.zeros((self.size, self.dimension))
        self.p_best = np.zeros((self.size, self.dimension))
        self.g_best = np.zeros((1, self.dimension))[0]
        self.x_test = x_test
        self.x_train = x_train
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]
            x_test1 = np.insert(self.x_test, 1, self.p_best[i])

            fit = self.fitness(x_test1)
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, x):
        x = [x]
        x = pd.DataFrame(x)

        std_x.fit_transform(self.x_train)
        x = std_x.transform(x)

        y = loaded_model.predict(x)
        result_x.append(x[0].tolist())

        result_y.append(y[0])
        return y

    def update(self, size):
        c1 = 2.0
        c2 = 2.0
        w = 0.8
        for i in range(size):
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            self.x[i] = self.x[i] + self.v[i]
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # std_x.fit_transform(self.x_train)
            x_test1 = np.insert(self.x_test, 1, self.x[i])
            # x_test1 = std_x.transform(x_test1)
            x_test2 = np.insert(self.x_test, 1, self.p_best[i])
            # x_test2 = std_x.transform(x_test2)
            x_test3 = np.insert(self.x_test, 1, self.g_best)
            # x_test3 = std_x.transform(x_test3)
            if self.fitness(x_test1) > self.fitness(x_test2):
                self.p_best[i] = self.x[i]
            if self.fitness(x_test1) > self.fitness(x_test3):
                self.g_best = self.x[i]

    def pso(self):

        best = []
        self.final_best = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0])
        for gen in range(self.time):
            self.update(self.size)
            x_test1 = np.insert(self.x_test, 1, self.g_best)
            # std_x.fit_transform(self.x_train)
            # x_test1 = std_x.transform(x_test1)
            x_test2 = np.insert(self.x_test, 1, self.final_best)
            # x_test2 = std_x.transform(x_test2)
            if self.fitness(x_test1) > self.fitness(x_test2):
                self.final_best = self.g_best.copy()
            print('best location：{}'.format(self.final_best))
            x_test3 = np.insert(self.x_test, 1, self.final_best)
            # x_test3 = std_x.transform(x_test3)
            temp = self.fitness(x_test3)
            print('：{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        plt.figure()
        plt.plot(t, best, color='red', marker='.', ms=15)
        plt.margins(0)
        plt.xlabel("Epoch time")
        plt.ylabel("logk")
        plt.title("")
        plt.show()


if __name__ == '__main__':
    data = load_data("data/result/XGBoost_PI_score.xlsx")
    data = data.values
    filter_index = []
    for i in range(len(data)):
        if data[i][1] > 0:
            filter_index.append(i)
    x_train, y_train, x_test, y_test, columns = set_all_feature_data()
    x_test = pd.DataFrame(x_test)
    x_test = x_train.iloc[49:50, filter_index]
    # x_test = x_test.iloc[0:1, filter_index]
    x_train = pd.DataFrame(x_train)
    x_train = x_train.iloc[:, filter_index]
    x_test = x_test.values
    print(x_test)
    x_test = x_test[0]

    x_test = np.delete(x_test, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    time = 50
    size = 50
    dimension = 11
    v_low = -2
    v_high = 2
    low = [0.77, 5, 5, 0.5, 80, 0.2, 0, 0, 0.002, 8.5, 2]
    up = [0.77, 10, 10, 100, 500, 4, 0, 0, 1, 90, 12]
    pso = PSO(dimension, time, size, low, up, v_low, v_high, x_test, x_train)
    pso.pso()
    result_x = std_x.inverse_transform(result_x)
    result_x = pd.DataFrame(result_x)
    result_x["new"] = result_y
    writer = pd.ExcelWriter("data/result/PSO_erlvbenfen.xlsx")
    pd.DataFrame(result_x).to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()