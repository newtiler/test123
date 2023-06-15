import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

anime_df = pd.read_csv('anime_with_synopsis.csv')
anime_df = anime_df.rename({'sypnopsis': 'Synopsis'}, axis=1)
anime_df = anime_df.rename({'MAL_id': 'anime_id'}, axis=1)

anime_copy = anime_df.copy()
anime_copy = anime_copy[['Name', 'Genres', 'Synopsis']]
anime_copy['tagline'] = anime_copy['Synopsis'] + anime_copy['Genres']
st.write(anime_copy)
