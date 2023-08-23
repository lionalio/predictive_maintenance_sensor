import sys
sys.path.append('../../src/')

from libs import *
from config import *
from data_processing import *
import streamlit as st

model = keras.models.load_model(PATH_SAVED_MODEL)
scaler = joblib.load(PATH_SCALER)

params = dict(
    period=30
)

def func_predict(input_data, features, label):
    X = data_transform(
        input_data, features, label, 
        scaler, params['period'], is_label=False
        )
    
    preds = model.predict(X)
    classes = np.argmax(preds, axis = 1)
    #classes = model.predict(X, verbose=1, batch_size=128)
    for i in range(X.shape[0]):
        status = 'OK' if classes[i] == 0 else 'Need reparing'
        st.write('predicting machine {} status: {}'.format(i, status))


def main_predict_page():
    st.title("Predict from input file")
    #url_data = st.input_text('url of data: ')
    #url_label = st.input_text('url of label: ')
    #url_meta = st.input_text('url of meta:')


    uploaded_file = st.file_uploader("Choose a historical file")
    if uploaded_file is not None:
        input = pd.read_csv(uploaded_file)
        #st.write(input)
        prediction(input, features, targets[0])


if __name__ == "__main__":
    main_predict_page()
