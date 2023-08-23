import sys
sys.path.append('../src/')

from libs import *
from config import *
from data_processing import *

import streamlit as st


model = keras.models.load_model(PATH_SAVED_MODEL)
scaler = joblib.load(PATH_SCALER)

params = dict(
    period=30
)


def training():
    pass


def main():
    st.title("Predictive Maintenance Home Page")
    st.sidebar.success("You are viewing the main page")

if __name__ == '__main__':
    main()