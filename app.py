import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



st.set_page_config(layout="wide")
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



with st.sidebar:
    st.image("logo.png")
    st.title('Auto ML')
    choice = st.radio("Navigation", ["Upload", "Pandas Profiling", "Generate Model", "Download", "Load Model"])
    st.info("Auto ML helps you to explore and build optimal models from data in just one click.")

if choice == "Upload":
    st.title("Upload Your Data for Modeling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        data = pd.read_csv(file, index_col=None)
        data.to_csv('sourcedata.csv', index=None)
        st.dataframe(data)

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Pandas Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Generate Model":
    st.title("Building Various Models")
    target = st.selectbox("Select your target from data", df.columns)
    model_type = st.selectbox("Select type of problem", ["Regression", "Classification"])
    if model_type == "Regression":
        from pycaret.regression import setup, compare_models, pull, save_model, load_model
    if model_type == "Classification":
        from pycaret.classification import setup, compare_models, pull, save_model, load_model

    if st.button("Run Model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML Experiment Settings")
        st.dataframe(setup_df)

        best_model = compare_models()
        compare_df = pull()
        st.info("This is ML Models")
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    st.title("Download the Model")
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the file", f, "trained_model.pkl")

if choice == "Load Model":
    st.title("Loading the Model")
    model_type = st.selectbox("Select type of problem", ["Regression", "Classification"])
    if model_type == "Regression":
        from pycaret.regression import load_model
    if model_type == "Classification":
        from pycaret.classification import load_model
    if st.button("Load"):
        pipeline = load_model("trained_model")
        st.write(pipeline)