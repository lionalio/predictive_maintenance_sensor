from libs import *
from read_data import *


def get_scaling(X_train, features, namesave):
    if os.path.isfile(namesave):
        print('Scaler has already existed. Loading...')
        scaler = joblib.load(namesave)
        return scaler
    
    print('Scaler not yet trained. Creating one from train data')
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
    #lstm_array=[]
    for start, stop in zip(range(0, n_elements - window), range(window, n_elements)):
        #if start % 10 == 0:
        #    print(start, stop)
        yield data_matrix[start:stop, :]
        #lstm_array.append(data_matrix[start:stop, :])

    #return np.array(lstm_array)


def get_labeling(y, window, label):
    data_matrix = y[label].values
    num_elements = data_matrix.shape[0]
    # Remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previuus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target. 
    return data_matrix[window:num_elements, :]


def data_transform(data, features, label, scaler, window, is_label=False):
    if not is_label:
        data[features] = scaler.transform(data[features])
        gen_data = [
            list(get_sequence(data[data['id'] == idx], window)) 
            for idx in data['id'].unique()
            ]
    else:
        gen_data = [
            list(get_labeling(data[data['id'] == idx], window, [label]))
            for idx in data['id'].unique()
        ]
    
    # Remove the zero indices (non-exist index)
    final_list = []
    for d in gen_data:
        if len(d) == 0:
            continue
        final_list.append(d)

    data_final = np.concatenate(list(final_list)).astype(np.float32)

    return data_final


if __name__ == '__main__':
    train, test = load_data(PATH_TRAIN, PATH_TEST)

    scaler = get_scaling(train, features, PATH_SCALER)

    X_train = data_transform(train, features, label, scaler, period, is_label=False)
    y_train = data_transform(train, features, label, scaler, period, is_label=True)
    X_test = data_transform(test, features, label, scaler, period, is_label=False)
    y_test = data_transform(test, features, label, scaler, period, is_label=True)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    with open(os.path.join(PATH_RAW_DATA, '/X_train.pkl'), 'wb') as f1:
        pkl.dump(X_train, f1)
    with open(os.path.join(PATH_RAW_DATA, '/y_train.pkl'), 'wb') as f2:
        pkl.dump(y_train, f2)
    with open(os.path.join(PATH_RAW_DATA, '/X_test.pkl'), 'wb') as f3:
        pkl.dump(X_test, f3)
    with open(os.path.join(PATH_RAW_DATA, '/y_test.pkl'), 'wb') as f4:
        pkl.dump(y_test, f4)