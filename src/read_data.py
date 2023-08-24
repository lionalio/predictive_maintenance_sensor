from multiprocessing import process
from libs import *
from config import *


def process_columns(data):
    data.drop([26, 27], axis=1, inplace=True)
    data.columns = columns

    rul = pd.DataFrame(data.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max_cycle']

    data = data.merge(rul, on=['id'], how='left')
    data['RUL'] = data['max_cycle'] - data['cycle']
    data[label] = data['RUL'].apply(lambda x: 1 if x <= period else 0)
    data.drop('max_cycle', axis=1, inplace=True)

    return data


def load_data(path_train, path_test):
    train = pd.read_csv(path_train, sep=' ', header=None)
    test = pd.read_csv(path_test, sep=' ', header=None)
    train = process_columns(train)
    test = process_columns(test)

    return train, test


if __name__ == '__main__':
    train, test = load_data(PATH_TRAIN, PATH_TEST)
    print(train.head())
    print(test.head())