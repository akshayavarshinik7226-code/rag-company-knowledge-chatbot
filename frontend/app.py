import streamlit as st
import requests

st.title("Company Knowledge Chatbot")

if st.button("Test Backend"):
    response = requests.get("http://127.0.0.1:8000/ping")

    if response.status_code == 200:
        st.success(response.json()["status"])
    else:
        st.error("Backend is not responding")