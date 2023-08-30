import sys
sys.path.append('../../src/')
sys.path.append('../')
from libs import *
from config import *
from data_processing import *
import streamlit as st
from app import func_predict

#model = mlflow.keras.load_model(PATH_SAVED_MODEL)
#scaler = joblib.load(PATH_SCALER)

params = dict(
    period=50
)



def main_predict_page():
    st.title("Predict from input file")
    #url_data = st.input_text('url of data: ')
    #url_label = st.input_text('url of label: ')
    #url_meta = st.input_text('url of meta:')


    uploaded_file = st.file_uploader("Choose a file for prediction")
    if uploaded_file is not None:
        input = pd.read_csv(uploaded_file, sep=' ', header=None)
        input = process_columns(input)
        #st.write(input)
        func_predict(input, features, label)


if __name__ == "__main__":
    main_predict_page()
