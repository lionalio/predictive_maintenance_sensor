from libs import *
from config import *


def load_data(path_train, path_test):
    train = pd.read_csv(path_train, sep=' ', header=None)
    train.drop([26, 27], axis=1, inplace=True)
    test = pd.read_csv(path_test, sep=' ', header=None)
    test.drop([26, 27], axis=1, inplace=True)
    train.columns = columns
    test.columns = columns

    rul = pd.DataFrame(train.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max_cycle']

    train = train.merge(rul, on=['id'], how='left')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train.drop('max_cycle', axis=1, inplace=True)
    train[label] = train['RUL'].apply(lambda x: 1 if x <= period else 0)

    test = test.merge(rul, on=['id'], how='left')
    test['RUL'] = test['max_cycle'] - test['cycle']
    test[label] = test['RUL'].apply(lambda x: 1 if x <= period else 0)
    test.drop('max_cycle', axis=1, inplace=True)

    return train, test


if __name__ == '__main__':
    train, test = load_data(PATH_TRAIN, PATH_TEST)
    print(train.head())
    print(test.head())