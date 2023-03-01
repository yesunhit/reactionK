
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessing:

    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.std_x = StandardScaler()
        self.std_y = StandardScaler()

    def load_data(self, data_path):
        descriptor = pd.read_excel(data_path, header=0)
        return descriptor

    def get_all_feature_data(self):
        data_train = self.load_data(self.train_data_path)
        data_test = self.load_data(self.test_data_path)
        x_train = data_train.iloc[:, 1:data_train.shape[1]]
        columns = x_train.columns
        y_train = data_train.iloc[:, 0]
        x_test = data_test.iloc[:, 1:data_train.shape[1]]
        y_test = data_test.iloc[:, 0]
        x_train = self.std_x.fit_transform(x_train)
        x_test = self.std_x.transform(x_test)
        return x_train, y_train, x_test, y_test, columns

    def get_filter_new_index_data(self, new_index):
        x_train, y_train, x_test, y_test, columns = self.get_all_feature_data()
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        x_train = x_train.iloc[:, new_index]
        x_test = x_test.iloc[:, new_index]
        x_train.columns = columns[new_index]
        x_test.columns = columns[new_index]
        x_train = x_train.values
        x_test = x_test.values
        return x_train, y_train, x_test, y_test, columns