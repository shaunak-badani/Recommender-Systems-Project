import pandas as pd
from pathlib import Path
import numpy as np

class NaiveRecommender:
    """
    Recommends restaurants based on the highest-rated ones in the cities where the user has previously reviewed businesses.
    """

    def __init__(self):
        self.reviews = None
        self.businesses = None

    def load_data(self):
        """Loads review and business data from JSON files."""
        datapath = Path("../../data")
        chunk_size = 10000  # Adjust based on memory constraints

        # Load reviews
        review_chunks = []
        try:
            for chunk in pd.read_json(datapath / "yelp_academic_dataset_review.json", lines=True, chunksize=chunk_size):
                review_chunks.append(chunk[['user_id', 'business_id']])
            self.reviews = pd.concat(review_chunks, ignore_index=True)
        except FileNotFoundError:
            print(f"Error: Review data file not found at {datapath / 'yelp_academic_dataset_review.json'}")
            self.reviews = pd.DataFrame(columns=['user_id', 'business_id']) # Empty DataFrame

        # Load busineses
        business_chunks = []
        try:
            for chunk in pd.read_json(datapath / "yelp_academic_dataset_business.json", lines=True, chunksize=chunk_size):
         
                required_cols = ['business_id', 'name', 'address', 'city', 'state', 'stars', 'review_count']
                # Handle potential missing 'is_open' column
                if 'is_open' in chunk.columns:
                    required_cols.append('is_open')
                
                # Select only existing required columns
                cols_to_select = [col for col in required_cols if col in chunk.columns]
                business_chunks.append(chunk[cols_to_select])

            self.businesses = pd.concat(business_chunks, ignore_index=True)
            # Pre-filter for open businesses if 'is_open' exists and was loaded
            if 'is_open' in self.businesses.columns:
                 self.businesses = self.businesses[self.businesses['is_open'] == 1].copy()
                 # Drop is_open after filtering as it might not be needed for recommendations
                 self.businesses = self.businesses.drop(columns=['is_open'])

        except FileNotFoundError:
            print(f"Error: Business data file not found at {datapath / 'yelp_academic_dataset_business.json'}")
            # Ensure schema matches even on error
            self.businesses = pd.DataFrame(columns=['business_id', 'name', 'address', 'city', 'state', 'stars', 'review_count']) # Empty DataFrame
        except Exception as e:
             print(f"An error occurred during business data loading: {e}")
             self.businesses = pd.DataFrame(columns=['business_id', 'name', 'address', 'city', 'state', 'stars', 'review_count'])


    def generate_recommendations(self, user_id, n=10, min_review_count=5):
        """
        Generates top N restaurant recommendations for a given user.

        Args:
            user_id (str): The ID of the user for whom to generate recommendations.
            n (int): The number of recommendations to return.
            min_review_count (int): Minimum number of reviews a business must have to be considered.

        Returns:
            list: A list of top N recommended business IDs, or an empty list if no recommendations can be made.
        """
        if self.reviews is None or self.businesses is None:
            print("Error: Data not loaded. Call load_data() first.")
            return []
        if self.reviews.empty or self.businesses.empty:
             print("Error: Dataframes are empty. Cannot generate recommendations.")
             return []

        # Find businesses reviewed by the user
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        if user_reviews.empty:
            print(f"User {user_id} has no reviews.")
            return []

        reviewed_business_ids = user_reviews['business_id'].unique()

        # Find the cities/states of these businesses
        reviewed_businesses = self.businesses[self.businesses['business_id'].isin(reviewed_business_ids)]
        user_locations = reviewed_businesses[['city', 'state']].drop_duplicates()

        if user_locations.empty:
             print(f"Could not determine locations for user {user_id}'s reviewed businesses.")
             return []

        # Filter all businesses to those in the users frequented locations
        candidate_restaurants = pd.merge(self.businesses, user_locations, on=['city', 'state'], how='inner')
        
        # Apply minimum review count filter
        if 'review_count' in candidate_restaurants.columns:
             candidate_restaurants = candidate_restaurants[candidate_restaurants['review_count'] >= min_review_count]

        # Sort by stars (desc) and review_count (desc)
        top_restaurants = candidate_restaurants.sort_values(
            by=['stars', 'review_count'],
            ascending=[False, False]
        )

        # Exclude restaurants the user has already reviewed
        top_restaurants = top_restaurants[~top_restaurants['business_id'].isin(reviewed_business_ids)]
        top_n_details = top_restaurants.head(n)
        recommendations_list = top_n_details.replace({np.nan: None}).to_dict('records')

        # only return list of business_ids
        return [restaurant['business_id'] for restaurant in recommendations_list]


if __name__ == '__main__':
    recommender = NaiveRecommender()
    recommender.load_data()

    test_user_id = "qVc8ODYU5SZjKXVBgXdI7w" # Example user
    recommendations = recommender.generate_recommendations(test_user_id)
    print(f"Recommendations for {test_user_id}:")
    print(recommendations)

    # Test user with reviews but maybe only in one/two place
    user_reviews = recommender.reviews['user_id'].value_counts()
    user_with_few_reviews = user_reviews[user_reviews < 3].index[0]
    recommendations_few = recommender.generate_recommendations(user_with_few_reviews)
    print(f"Recommendations for user with few reviews ({user_with_few_reviews}):")
    print(recommendations_few) 