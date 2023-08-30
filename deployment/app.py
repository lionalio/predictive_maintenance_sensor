import sys
sys.path.append('../src/')

from libs import *
from config import *
from data_processing import *

import streamlit as st


@st.cache_resource
def load_models():
    model = mlflow.keras.load_model(PATH_SAVED_MODEL)
    scaler = joblib.load(PATH_SCALER)    
    return model, scaler


model, scaler = load_models()

params = dict(
    period=50
)

def func_predict(input_data, features, label):
    #model = Models.model
    #scaler = Models.scaler
    X = data_transform(
        input_data, features, label, 
        scaler, params['period'], is_label=False
        )
    
    #preds = model.predict(X)
    probs = model.predict(X, verbose=1).reshape(-1)
    classes = np.array([0 if p <= 0.5 else 1 for p in probs])
    for row, pred in zip(X[:, 0], classes):
        status = 'OK' if pred == 0 else 'Need reparing'
        st.write('predicting machine {} status: {}'.format(row[0], status))


def main():
    st.title("Predictive Maintenance Home Page")
    st.sidebar.success("You are viewing the main page")

    #st.title("Upload for training-later")

    #st.title("Predict from input file")
    #url_data = st.input_text('url of data: ')
    #url_label = st.input_text('url of label: ')
    #url_meta = st.input_text('url of meta:')


    #uploaded_file = st.file_uploader("Choose a file for prediction")
    #if uploaded_file is not None:
    #    input = pd.read_csv(uploaded_file, sep=' ', header=None)
    #    input = process_columns(input)
        #st.write(input)
    #    func_predict(input, features, label)


if __name__ == '__main__':
    main()