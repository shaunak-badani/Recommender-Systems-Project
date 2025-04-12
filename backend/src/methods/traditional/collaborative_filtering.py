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


    def generate_recommendations(self, user_id, k=10, n_top_items=5):
        """
        Generates recommendations using multiple top-rated items from the user.
        
        Args:
            user_id (str): The user to generate recommendations for
            k (int): Number of recommendations to return
            n_top_items (int): Number of user's top items to consider
        """
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        if user_ratings.empty:
            return []
        
        # Get multiple top-rated items instead of just one
        user_ratings = user_ratings.sort_values(by='stars', ascending=False)
        top_rated_ids = user_ratings.head(n_top_items)['business_id'].tolist()
        top_rated_with_embeddings = [item_id for item_id in top_rated_ids 
                                if item_id in self.restaurant_embeddings.index]
        
        if not top_rated_with_embeddings:
            return []
        
        # Calculate combined similarity across all top items
        combined_sim_series = None
        
        for item_id in top_rated_with_embeddings:
            try:
                target_embedding = self.restaurant_embeddings.loc[item_id].values
                all_embeddings = self.restaurant_embeddings.values
                similarities = (all_embeddings @ target_embedding[:, np.newaxis]).flatten()
                sim_series = pd.Series(similarities, index=self.restaurant_embeddings.index)
                
                # Combine similarities
                if combined_sim_series is None:
                    combined_sim_series = sim_series
                else:
                    combined_sim_series += sim_series
                    
            except Exception as e:
                print(f"Error calculating similarity for {item_id}: {e}")
                continue
        
        if combined_sim_series is None:
            return []
        
        # Exclude items the user already rated
        rated_items = user_ratings['business_id'].unique()
        combined_sim_series = combined_sim_series.drop(rated_items, errors='ignore')
        
        # Get top k similar items
        recommended_ids = combined_sim_series.nlargest(k).index.tolist()
        
        return recommended_ids