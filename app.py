import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

anime_df = pd.read_csv('anime_with_synopsis.csv')
st.write(anime_df)
