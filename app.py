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

tfidf = TfidfVectorizer(stop_words='english')
anime_copy['tagline'] = anime_copy['tagline'].fillna('')
tfidf_matrix = tfidf.fit_transform(anime_copy['tagline'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(Name, cosine_sim=cosine_sim):
    idx = anime_copy[anime_copy['Name'] == Name].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), reverse=True, key=lambda x: x[1])[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return anime_df[['Name', 'Synopsis']].iloc[anime_indices]
st.write(anime_copy)
