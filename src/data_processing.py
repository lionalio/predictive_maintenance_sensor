from libs import *
from read_data import *


def get_scaling(X_train, features, namesave):
    scaler = MinMaxScaler()
    scaler.fit(X_train[features])
    joblib.dump(scaler, namesave)

    return scaler


def get_sequence(X, window):
    data_matrix = X.values
    n_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 141 191 -> from row 141 to 191
    for start, stop in zip(range(0, n_elements-window), range(window, n_elements)):
        yield data_matrix[start:stop, :]


def get_labeling(y, window, label):
    data_matrix = y[label].values
    num_elements = data_matrix.shape[0]
    # Remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previuus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target. 
    return data_matrix[window:num_elements, :]


def create_train_test(X, y, ids, split, timeline):
    pass


def data_transform(data, features, label, scaler, window, is_label=False):
    if not is_label:
        data[features] = scaler.transform(data[features])
        gen_data = [
            list(get_sequence(data[data['machineID'] == idx], window)) 
            for idx in data['machineID'].unique()
            ]
    else:
        gen_data =[
            list(get_labeling(data[data['machineID'] == idx], window, [label]))
            for idx in data['machineID'].unique()
        ]

    data_final = np.concatenate(list(gen_data)).astype(np.float32)

    return data_final


if __name__ == '__main__':
    pass