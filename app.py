import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# OMDb API key
API_KEY = '5f53ef7b'

# Function to fetch movie data (poster, plot, and more)
def fetch_movie_data(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        poster = data.get('Poster', 'https://via.placeholder.com/150')
        plot = data.get('Plot', 'Description not available')
        year = data.get('Year', 'N/A')
        actors = data.get('Actors', 'N/A')
        director = data.get('Director', 'N/A')
        budget = data.get('BoxOffice', 'N/A') 
        return poster, plot, year, actors, director, budget
    except:
        return 'https://via.placeholder.com/150', 'Description not available', 'N/A', 'N/A', 'N/A', 'N/A'


# Load dataset
movies = pd.read_csv("imdb_top_1000.csv")

# Fill NaNs and combine relevant columns
movies.fillna('', inplace=True)
movies['combined_features'] = (
    movies['Genre'] + ' ' +
    movies['Director'] + ' ' +
    movies['Star1'] + ' ' +
    movies['Star2'] + ' ' +
    movies['Star3'] + ' ' +
    movies['Star4'] + ' ' +
    movies['Overview']
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
vector_matrix = tfidf.fit_transform(movies['combined_features'])

# Cosine similarity
similarity = cosine_similarity(vector_matrix)


# Recommend function
def recommend(movie_title):
    movie_title = movie_title.lower()
    indices = movies[movies['Series_Title'].str.lower() == movie_title].index

    if len(indices) == 0:
        return []

    idx = indices[0]
    similarity_scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_movies = sorted_scores[1:6]
    
    recommendations = []
    for i in top_movies:
        movie_data = {}
        title = movies.iloc[i[0]]['Series_Title']
        poster, plot, year, actors, director, budget = fetch_movie_data(title)
        movie_data['title'] = title
        movie_data['poster'] = poster
        movie_data['plot'] = plot
        movie_data['year'] = year
        movie_data['actors'] = actors
        movie_data['director'] = director
        movie_data['budget'] = budget
        recommendations.append(movie_data)

    return recommendations


# Streamlit UI
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎬"
)

# Custom CSS for a better look and feel and for increased spacing
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stTextInput>div>div>input {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
    }
    .stSelectbox>div>div {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
    }
    h1, h3 {
        color: #58a6ff;
    }
    .flex-container {
        display: flex;
        align-items: flex-start;
        gap: 20px;
        padding: 10px;
    }
    .movie-details p {
        margin: 0;
        padding: 0;
    }
    .vertical-separator {
        border-right: 1px solid #30363d;
        padding-right: 20px;
    }
    /* New style to increase space between rows */
    hr {
        margin-top: 40px;
        margin-bottom: 40px;
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)


st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie you like:", movies['Series_Title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if recommendations:
        for i in range(0, len(recommendations), 2):
            col1, col2 = st.columns(2)

            with col1:
                rec1 = recommendations[i]
                st.markdown(f"""
                <div class="vertical-separator">
                    <h3>{rec1['title']} ({rec1['year']})</h3>
                    <div class="flex-container">
                        <img src="{rec1['poster']}" style="width: 150px; height: auto;">
                        <div>
                            <p style="margin-top:0;"><strong><span style="color:#58a6ff;">Director:</span></strong> {rec1['director']}</p>
                            <p><strong><span style="color:#ffa658;">Cast:</span></strong> {rec1['actors']}</p>
                            <p><strong><span style="color:#5cb85c;">Budget:</span></strong> {rec1['budget']}</p>
                            <br>
                            <p style="text-align: justify;">{rec1['plot']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if i + 1 < len(recommendations):
                with col2:
                    rec2 = recommendations[i+1]
                    st.markdown(f"""
                    <h3>{rec2['title']} ({rec2['year']})</h3>
                    <div class="flex-container">
                        <img src="{rec2['poster']}" style="width: 150px; height: auto;">
                        <div>
                            <p style="margin-top:0;"><strong><span style="color:#58a6ff;">Director:</span></strong> {rec2['director']}</p>
                            <p><strong><span style="color:#ffa658;">Cast:</span></strong> {rec2['actors']}</p>
                            <p><strong><span style="color:#5cb85c;">Budget:</span></strong> {rec2['budget']}</p>
                            <br>
                            <p style="text-align: justify;">{rec2['plot']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
    else:
        st.warning("Movie not found in database. Please try another one.")