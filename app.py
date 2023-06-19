import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
#import scipy as sp

anime_df = pd.read_csv('anime1.csv')
rating_df = pd.read_csv('rating1.csv')

anime_copy = anime_df.copy()
anime_copy = anime_copy[['Name', 'Genres', 'Synopsis']]
anime_copy['tagline'] = anime_copy['Synopsis'] + anime_copy['Genres']

tfidf = TfidfVectorizer(stop_words='english')
anime_copy['tagline'] = anime_copy['tagline'].fillna('')
tfidf_matrix = tfidf.fit_transform(anime_copy['tagline'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(Name, topn, cosine_sim=cosine_sim):
    idx = anime_copy[anime_copy['Name'] == Name].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), reverse=True, key=lambda x: x[1])[1:topn+1]
    anime_indices = [i[0] for i in sim_scores]
    return anime_df[['Name', 'Synopsis']].iloc[anime_indices]

new_rating_df = rating_df.copy()

cmat = pd.crosstab(new_rating_df['user_id'],new_rating_df['anime_id'],new_rating_df['rating'],aggfunc=sum)
cmat = cmat.fillna(0)

from sklearn.decomposition import NMF
nmf = NMF(100)
nmf.fit(cmat)

H = pd.DataFrame(np.round(nmf.components_, 2), columns=cmat.columns)
W = pd.DataFrame(np.round(nmf.transform(cmat), 2), columns=H.index)
recommend = pd.DataFrame(np.round(np.dot(W, H), 2), columns=H.columns)
recommend.index = cmat.index

def recommendation(uid,topn):
    res = list(recommend.iloc[uid].sort_values(ascending=False)[0:topn].index)
    res = anime_df[anime_df['anime_id'].isin(res)]
    res = res[['Name','Synopsis']]
    return res

def main():
    st.title(':red[Anime] Recommendation System :movie_camera:')
    anime_list = anime_df['Name'].values
    types = ['--Select--', 'Content-Based', 'Collaborative Filtering']
    type_sel = st.selectbox('Select Recommendation Type', types)
    if type_sel == types[0]:
        st.warning('Please select Recommendation Type!!')
    elif type_sel == types[1]:
        def Table(df):
            fig=go.Figure(go.Table(columnorder = [1,2,3],
                columnwidth = [1000,1000],
                header=dict(values=['Name','Synopsis'],
                    line_color='black',font=dict(color='white',size= 19),height=40,
                    fill_color='red',
                    align=['center','center']),
                cells=dict(values=[df.Name,df.Synopsis],
                    fill_color='#ffdac4',line_color='grey',
                    font=dict(color='black', family="Lato", size=16),
                    align='left')))
            fig.update_layout(height=1200, title ={'text': "This is a recommendation for you", 'font': {'size': 22}})
            return st.plotly_chart(fig,use_container_width=True)
        selected_anime = st.selectbox("Type or select an anime from the dropdown",anime_list)
        topn = round(st.number_input("Select number of recommendations",min_value=5, max_value=20, step=1))
        if st.button('Show Recommendation'):
            recommended_anime_names = get_recommendations(selected_anime, topn)
            #list_of_recommended_anime = recommended_anime_names.to_list()
        # st.write(recommended_anime_names[['title', 'description']])
            Table(recommended_anime_names)
    elif type_sel == types[2]:
        def Table2(df2):
            fig=go.Figure(go.Table(columnorder = [1,2,3],
                columnwidth = [1000,1000],
                header=dict(values=['Name','Synopsis'],
                    line_color='black',font=dict(color='white',size= 19),height=40,
                    fill_color='red',
                    align=['center','center']),
                cells=dict(values=[df2.Name,df2.Synopsis],
                    fill_color='#ffdac4',line_color='grey',
                    font=dict(color='black', family="Lato", size=16),
                    align='left')))
            fig.update_layout(height=1200, title ={'text': "This is a recommendation for you", 'font': {'size': 22}})
            return st.plotly_chart(fig,use_container_width=True)
        selected_anime_fuser = round(st.number_input("Select user id that you want",min_value=recommend.index[0], max_value=recommend.index[-1]))
        topn = round(st.number_input("Select number of recommendations",min_value=5, max_value=20, step=1))
        if st.button('Show Recommendation for user'):
            recommended_anime_fuser = recommendation(selected_anime_fuser, topn)
        #list_of_recommended_anime = recommended_anime_fuser.to_list()
        # st.write(recommended_anime_fuser[['title', 'description']])
            Table2(recommended_anime_fuser)
main()  

st.write('  '
         )
st.write(' ')
