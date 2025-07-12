import streamlit as st
import pickle

@st.cache_data
def cache_data(data):
    return data

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
