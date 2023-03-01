import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    descriptor = pd.read_excel(file_path, header=0)
    return descriptor


data = load_data('../data/data_with_reaction_conditions.xlsx')
# data = load_data('../data/data_without_reaction_conditions.xlsx')

train_data, test_data = train_test_split(data, test_size=0.1, random_state=1)

writer = pd.ExcelWriter('../data/train_data.xlsx')
pd.DataFrame(train_data).to_excel(writer, sheet_name='sheet1', index=False)
writer.save()

writer2 = pd.ExcelWriter('../data/test_data.xlsx')
pd.DataFrame(test_data).to_excel(writer2, sheet_name='sheet1', index=False)
writer2.save()
