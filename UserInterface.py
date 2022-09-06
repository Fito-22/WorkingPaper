from turtle import pd
import streamlit as st

import numpy as np
import pandas as pd
import requests
from pdfminer.high_level import extract_text

st.markdown('Hello World')
file=st.file_uploader('Upload a file')

text = extract_text(file)
st.markdown(text)

url='https://workingpaper-x6wjgko6ta-ew.a.run.app/pred'
params={'file': text}

response=requests.get(url, params)
st.markdown(response.content)
#print(response.content)
