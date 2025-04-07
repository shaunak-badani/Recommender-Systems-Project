import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path



class CollaborativeRecommender:
    """
    Class for implementation of the item-item method for collaborative filtering
    """

    # def __init__(self, ratings, restaurants, restaurant_embeddings):
    #     self.ratings = ratings
    #     self.restaurants = restaurants
    #     self.restaurant_embeddings = restaurant_embeddings


    def generate_recommendations(self, user):
        """
        Args: user id e.g. qVc8ODYU5SZjKXVBgXdI7w
        Returns:
        top_rated_restaurant: e.g. w3giBYDmPWWnsNq5Sr2KQA
        most_similar_restaurant_ids: e.g. ['iZVfWpijwWyX_WR7hQyG9A', '-qjKoIo4tvWc6yF5DYVveg',
        'Kd1M6yXCpyhyqOYw-PPM6Q', 'tu2x5W3D7K1WMdtGi1J9Bw',
        'In47HN_pJzDdIyxYmDxCKw', 'vVN9HVQ_GTbfBt_Z0mS37w',
        '4T6snSpDCi0dQal8W_39zQ', '5vIOqHKIQWdmV-XNVmH0NQ',
        'JXvCRLxCDB5NbzRLoa6zWg', '3IC1K9FZ0Q1iMYMkHhkcBw']
        """
        # Get top rated restaurant by user
        user_exists = (self.ratings['user_id']==user).sum()
        if not user_exists:
            return None, [None]
        user_ratings = self.ratings.loc[self.ratings['user_id']==user]
        user_ratings = user_ratings.sort_values(by='stars',axis=0,ascending=False)
        top_rated_restaurant = user_ratings.iloc[0,:]['business_id']
        print("Top rated restaurant: ", top_rated_restaurant)
        # top_rated_restaurant_name = self.restaurants.loc[self.restaurants['business_id']==top_rated_restaurant,'name'].values[0]
        # Find most similar restaurants to the user's top rated restaurant
        cosine_similarity = (self.restaurant_embeddings.values @ self.restaurant_embeddings.loc['Pns2l4eNsfO8kk83dixA6A'].to_numpy()[:, np.newaxis]).flatten()
        top_indices_similar = np.argsort(cosine_similarity)[::-1]

        # Get 10 most similar movies excluding the movie itself
        most_similar = top_indices_similar[1:11]
        most_similar_restaurant_ids = self.restaurant_embeddings.index[most_similar].values
        return top_rated_restaurant, most_similar_restaurant_ids
    
    def load_data(self):
        chunk_size = 10000  # Adjust based on available memory
        datapath = Path("../../data/Yelp-JSON/Yelp-JSON/yelp_dataset")

        reviews = []  # List to store chunks
        for chunk in pd.read_json(datapath / "yelp_academic_dataset_review.json", lines=True, chunksize = chunk_size):
            reviews.append(chunk)

        reviews = pd.concat(reviews, ignore_index=True) 

        restaurants = []  # List to store chunks
        chunk_size = 10000  # Adjust based on available memory

        for chunk in pd.read_json(datapath / "yelp_academic_dataset_business.json", lines = True, chunksize = chunk_size):
            restaurants.append(chunk)

        restaurants = pd.concat(restaurants, ignore_index=True)
        restaurants['categories'] = restaurants['categories'].fillna('')
        tokenizer = lambda x: [cat.strip() for cat in x.split(',')]
        vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True)
        vectorized_restaurants = vectorizer.fit_transform(restaurants['categories'])
        vectorized_rest_df = pd.DataFrame(vectorized_restaurants.toarray().astype(np.float32), columns=vectorizer.get_feature_names_out())
        vectorized_rest_df.index = restaurants['business_id'].values
        vectorized_rest_df.index.name = None
        row_norms = np.linalg.norm(vectorized_rest_df.values, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        vectorized_rest_df[:] = vectorized_rest_df.values / row_norms
        ratings = reviews[['user_id', 'business_id', 'stars']]
        self.ratings = ratings
        self.restaurants = restaurants
        self.restaurant_embeddings = vectorized_rest_df



if __name__ == "__main__":
    chunk_size = 10000  # Adjust based on available memory
    datapath = Path("./data/Yelp-JSON/Yelp-JSON/yelp_dataset")

    reviews = []  # List to store chunks
    for chunk in pd.read_json(datapath / "yelp_academic_dataset_review.json", lines=True, chunksize = chunk_size):
        reviews.append(chunk)

    reviews = pd.concat(reviews, ignore_index=True) 

    restaurants = []  # List to store chunks
    chunk_size = 10000  # Adjust based on available memory

    for chunk in pd.read_json(datapath / "yelp_academic_dataset_business.json", lines = True, chunksize = chunk_size):
        restaurants.append(chunk)

    restaurants = pd.concat(restaurants, ignore_index=True)
    restaurants['categories'] = restaurants['categories'].fillna('')
    tokenizer = lambda x: [cat.strip() for cat in x.split(',')]
    vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True)
    vectorized_restaurants = vectorizer.fit_transform(restaurants['categories'])
    vectorized_rest_df = pd.DataFrame(vectorized_restaurants.toarray().astype(np.float32), columns=vectorizer.get_feature_names_out())
    vectorized_rest_df.index = restaurants['business_id'].values
    vectorized_rest_df.index.name = None
    row_norms = np.linalg.norm(vectorized_rest_df.values, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    vectorized_rest_df[:] = vectorized_rest_df.values / row_norms
    ratings = reviews[['user_id', 'business_id', 'stars']]
    # recommender = CollaborativeRecommender(ratings, restaurants, vectorized_rest_df)
    recommender = CollaborativeRecommender()
    recommendations = recommender.generate_recommendations("mh_-eMZ6K5RLWhZyISBhwA")
    print(recommendations)