import os
import geopandas as gpd
import pandas as pd
import math
from shapely.geometry import Point, Polygon
from itertools import product
#from google.colab import files
import io
import zipfile
#upload csv file
import streamlit as st
st.title("CSV File Uploader")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file using pandas
    df = pd.read_csv(uploaded_file)
    st.write("Fields in your CSV file:")
    st.write(list(df.columns))  # Display column names
    st.dataframe(df)            # Display the data as a table