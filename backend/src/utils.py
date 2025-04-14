from pathlib import Path
import pandas as pd
import os
import numpy as np
import base64
from sklearn.feature_extraction.text import CountVectorizer

def load_and_prep_data(datapath):
    """Loads and preprocesses review and business data.
       Returns tuple: (ratings_df, restaurants_df, restaurant_embeddings)
    """
    print("Loading review data...")
    try:
        reviews_df_full = pd.read_json(datapath / "yelp_academic_dataset_review.json", lines=True,
                                     dtype={'stars': 'float32'})
        print(f"Loaded {len(reviews_df_full)} reviews.")
        required_cols = ['user_id', 'business_id', 'stars']
        if not all(col in reviews_df_full.columns for col in required_cols):
            print(f"Error: Missing one or more required columns {required_cols} in the JSON review data.")
            raise ValueError("Missing required review columns")
        reviews_df = reviews_df_full[required_cols].copy()
        del reviews_df_full

        reviews_df.dropna(subset=['user_id', 'business_id', 'stars'], inplace=True)
        print(f"{len(reviews_df)} reviews after NaN drop.")
    except Exception as e:
        print(f"Error loading review data: {e}")
        raise

    print("Loading business data...")
    try:
        restaurants_df = pd.read_json(datapath / "yelp_academic_dataset_business.json", lines=True)
        print(f"Loaded {len(restaurants_df)} businesses.")
    except Exception as e:
        print(f"Error loading business data: {e}")
        raise

    print("Building restaurant category embeddings...")
    restaurants_df['categories'] = restaurants_df['categories'].fillna('')
    tokenizer = lambda x: [cat.strip() for cat in x.split(',')]
    vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True)
    vectorized_restaurants = vectorizer.fit_transform(restaurants_df['categories'])
    restaurant_embeddings = pd.DataFrame(vectorized_restaurants.toarray().astype(np.float32),
                                      columns=vectorizer.get_feature_names_out(),
                                      index=restaurants_df['business_id'])
    restaurant_embeddings.index.name = 'business_id'

    # Normalize embeddings
    row_norms = np.linalg.norm(restaurant_embeddings.values, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1 # Avoid division by zero
    restaurant_embeddings[:] = restaurant_embeddings.values / row_norms
    print("Embeddings built and normalized.")

    return reviews_df, restaurants_df, restaurant_embeddings
# --- End Helper --- #

class Utils:

    @classmethod
    def photo_id_to_image_json(cls, photo_id):
        """
        Given a photo_id, returns a base64 image JSON. Uses fallback image if photo_id is NaN or file not found.
        """
        datapath = Path("../../data")
        photos_dir = datapath / 'photos'
        fallback_image_path = datapath / "not-available.png"
        if photo_id is None or (isinstance(photo_id, float) and np.isnan(photo_id)):
            filename = fallback_image_path
            # photo_id = "fallback" 
        else:
            filename = os.path.join(photos_dir, f"{photo_id}.jpg")
            if not os.path.exists(filename):
                filename = fallback_image_path
                # photo_id = "fallback"

        with open(filename, "rb") as f:
            image_data = f.read()

        encoded = base64.b64encode(image_data).decode("utf-8")
        return encoded

    @staticmethod
    def gift_wrap_restaurant_data(restaurant_ids):
        """
        Args: resturant_ids: List of business_ids
        Returns: Array of JSON with restaurant names, addresses, and images
        """
        datapath = Path("../../data")
        restaurants = []

        try:
            restaurants = pd.read_json(datapath / "yelp_academic_dataset_business.json", lines=True)
        except Exception as e:
            print(f"Error loading business data in gift_wrap: {e}")
            return [] # Return empty list on error

        restaurants = restaurants[restaurants['business_id'].isin(restaurant_ids)]


        # Temporarily use fallback image for all until photo logic is confirmed/needed
        restaurants['image'] = (fallback_image_path := datapath / "not-available.png").read_bytes()
        restaurants['image'] = restaurants['image'].apply(lambda x: base64.b64encode(x).decode("utf-8"))

        columns_to_keep = ['business_id', 'name', 'address', 'image']
        # print(merged[columns_to_keep]) # Debug print
        return restaurants[columns_to_keep].to_dict(orient="records")

class PopularityRecommender:
    """
    Simple baseline recommender that recommends the most popular items
    (highest average rating and most reviews) that the user hasn't rated yet.
    """
    def __init__(self, ratings_df, min_review_count=5):
        self.ratings = ratings_df
        business_stats = self.ratings.groupby('business_id').agg(
            avg_rating=('stars', 'mean'),
            review_count=('user_id', 'count')
        )
        business_stats = business_stats[business_stats['review_count'] >= min_review_count]
        self.top_businesses = business_stats.sort_values(
            by=['avg_rating', 'review_count'], 
            ascending=[False, False]
        ).index.tolist()
        
    def generate_recommendations(self, user_id, k=10):
        user_rated = self.ratings[self.ratings['user_id'] == user_id]['business_id'].unique()
        recommendations = [bid for bid in self.top_businesses if bid not in user_rated][:k]
        return recommendations
