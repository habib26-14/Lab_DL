import streamlit as st
import json
import requests

st.title("Test API")
if st.button("testAP"):
    res = requests.get(url="http://127.0.0.1:8080/")
    st.subheader(res.text)