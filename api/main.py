import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

import sys
sys.path.append('../src/')

from libs import *
from config import *
from data_processing import *
from io import StringIO


scaler = joblib.load(PATH_SCALER)
params = dict(
    period=50
)

def func_predict(input_data, features, label):
    global model, scaler
    X = data_transform(
        input_data, features, label, 
        scaler, params['period'], is_label=False
        )
    
    #preds = model.predict(X)
    probs = model.predict(X, verbose=1).reshape(-1)
    classes = np.array([0 if p <= 0.5 else 1 for p in probs])
    results = {}
    i=0
    for row, pred in zip(X[:, 0], classes):
        status = 'OK' if pred == 0 else 'Need reparing'
        #print('predicting machine {} status: {}'.format(row[0], status))
        results[i] = 'machine {}: {}'.format(int(row[0]), status)
        i += 1

    return results


app = FastAPI()

@app.get('/')
async def index():
    return {"text":"Our First route"}


@app.post("/predict_uploadfile/")
async def predict_upload_file(file: UploadFile):
    data = pd.read_csv(StringIO(str(file.file.read(), 'utf-8')), sep=' ', header=None) #, encoding='utf-8')
    print(file.filename)
    print(data)
    input = process_columns(data)
    results = func_predict(input, features, label)

    return results


if __name__ == '__main__':
    model = mlflow.keras.load_model(PATH_SAVED_MODEL)
    uvicorn.run(app,host="127.0.0.1",port=8000)