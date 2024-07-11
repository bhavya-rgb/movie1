import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
df = pd.read_csv('/content/tmovies.csv', encoding='latin-1')

def recommend_movies(user_id, Genre):
    # Filter movies by genre, handling missing values
    genre_movies = df[df['Genre'].str.contains(Genre, na=False)]

    # Create a user-movie ratings matrix
    ratings_matrix = genre_movies.pivot_table(index='userid', columns='Movie', values='Rating').fillna(0)

    # Calculate cosine similarity between movies based on ratings
    cosine_sim = cosine_similarity(ratings_matrix.T)

    # Get index of the user
    user_index = ratings_matrix.index.get_loc(user_id)

    # Get similarity scores with other movies
    sim_scores = list(enumerate(cosine_sim[user_index]))

    # Sort movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 similar movies
    sim_scores = sim_scores[1:6]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return recommended movie titles
    recommended_movies = ratings_matrix.columns[movie_indices].tolist()

    return recommended_movies

# Streamlit UI
st.title('Movie Recommendation App')

# Input fields
user_id = st.number_input('Enter User ID', min_value=1, max_value=df['userid'].max(), value=11, step=1)
genre = st.text_input('Enter Genre', 'Action')

# Button to trigger recommendation
if st.button('Recommend Movies'):
    recommended_movies = recommend_movies(user_id, genre)
    st.subheader(f"Recommended movies for user {user_id} in genre '{genre}':")
    for movie in recommended_movies:
        st.write(f"- {movie}")
