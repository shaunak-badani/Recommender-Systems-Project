import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path



class CollaborativeRecommender:
    """
    Implementation of an item-item similarity method based on user's top-rated item.
    Uses category embeddings for similarity calculation.
    """

    # Modified __init__ to accept data directly
    def __init__(self, ratings_df, restaurant_embeddings_df):
        """
        Initializes the recommender with training ratings and pre-built embeddings.

        Args:
            ratings_df (pd.DataFrame): DataFrame containing user ratings (user_id, business_id, stars).
                                       Should typically be the *training* set for evaluation.
            restaurant_embeddings_df (pd.DataFrame): DataFrame where index is business_id and columns
                                                  are normalized category embeddings.
        """
        if not isinstance(ratings_df, pd.DataFrame) or not isinstance(restaurant_embeddings_df, pd.DataFrame):
            raise ValueError("ratings_df and restaurant_embeddings_df must be pandas DataFrames.")
        if not {'user_id', 'business_id', 'stars'}.issubset(ratings_df.columns):
             raise ValueError("ratings_df must contain 'user_id', 'business_id', 'stars' columns.")

        self.ratings = ratings_df
        if restaurant_embeddings_df.index.name != 'business_id':
             print("Warning: Setting restaurant_embeddings_df index name to 'business_id'")
             restaurant_embeddings_df.index.name = 'business_id'
        self.restaurant_embeddings = restaurant_embeddings_df


    def generate_recommendations(self, user_id, k=10):
        """
        Generates top-k recommendations for a given user.

        Args:
            user_id (str): The ID of the user to generate recommendations for.
            k (int): The number of recommendations to return.

        Returns:
            list: A list of top-k recommended business_ids, or an empty list if recommendations
                  cannot be generated.
        """
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]

        if user_ratings.empty:
            return [] # Return empty list if user has no ratings in training data

        user_ratings = user_ratings.sort_values(by='stars', ascending=False)
        top_rated_restaurant_id = user_ratings.iloc[0]['business_id']

        if top_rated_restaurant_id not in self.restaurant_embeddings.index:
            return [] # Return empty list if top item has no embedding

        try:
            # Calculate cosine similarity
            target_embedding = self.restaurant_embeddings.loc[top_rated_restaurant_id].values
            all_embeddings = self.restaurant_embeddings.values
            similarities = (all_embeddings @ target_embedding[:, np.newaxis]).flatten()
            sim_series = pd.Series(similarities, index=self.restaurant_embeddings.index)
        except Exception as e:
            print(f"Error calculating similarity for {top_rated_restaurant_id}: {e}")
            return []

        # Exclude the top-rated item itself
        sim_series = sim_series.drop(top_rated_restaurant_id, errors='ignore')

        # Exclude items the user already rated in the training set
        rated_in_train = user_ratings['business_id'].unique()
        sim_series = sim_series.drop(rated_in_train, errors='ignore')

        # Get top k similar items
        recommended_ids = sim_series.nlargest(k).index.tolist()

        return recommended_ids

