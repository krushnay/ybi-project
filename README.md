import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge the datasets on the movieId column
data = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0
user_item_matrix_filled = user_item_matrix.fillna(0)

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)

# Convert the similarity matrix to a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_recommendations(user_id, num_recommendations=5):
    # Get the user's similarity scores
    user_sim_scores = user_similarity_df[user_id]
    
    # Get the indices of the most similar users
    similar_users = user_sim_scores.sort_values(ascending=False).index[1:]
    
    # Get the movies rated by similar users
    similar_users_ratings = user_item_matrix.loc[similar_users]
    
    # Calculate the average rating for each movie
    movie_recommendations = similar_users_ratings.mean(axis=0).sort_values(ascending=False)
    
    # Filter out movies already rated by the user
    user_rated_movies = user_item_matrix.loc[user_id].dropna().index
    movie_recommendations = movie_recommendations.drop(user_rated_movies, errors='ignore')
    
    # Return the top recommendations
    return movie_recommendations.head(num_recommendations)

# Get recommendations for a specific user
user_id = 1
recommendations = get_recommendations(user_id)

print(f"Recommendations for User {user_id}:")
print(recommendations)
