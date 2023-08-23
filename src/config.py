import os

PATH_RAW_DATA = '../data/raw/'
PATH_PROCESSED_DATA = '../data/processed/'
PATH_MODEL = '../models/'

PATH_TRAIN = os.path.join(PATH_RAW_DATA, 'PM_train.txt')
PATH_TEST = os.path.join(PATH_RAW_DATA, 'PM_test.txt')


name_scaler = 'scaler.gz'
PATH_SCALER = os.path.join(PATH_MODEL, name_scaler)

name_model = 'transformer.keras'
PATH_SAVED_MODEL = os.path.join(PATH_MODEL, name_model)


columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 
            's1', 's2', 's3', 's4', 's5', 
            's6', 's7', 's8', 's9', 's10', 
            's11', 's12', 's13', 's14', 's15', 
            's16', 's17', 's18', 's19', 's20', 
            's21'
            ]

features = [f for f in columns if f not in ['id', 'cycle', 'setting1', 'setting2', 'setting3']]
label = 'label'
period=30