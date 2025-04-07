from pathlib import Path
import numpy as np
import pandas as pd
import os


def condense_data():
    """
    Takes all reviews from 1st March 2021, filters out reviews
    and stores them in data folder
    """
    datapath = Path("./raw-data/Yelp-JSON/Yelp-JSON/yelp_dataset")
    chunk_size = 10000  # Adjust based on available memory
    reviews = []  # List to store chunks
    for chunk in pd.read_json(datapath / "yelp_academic_dataset_review.json", lines=True, chunksize = chunk_size):
        reviews.append(chunk)
    reviews = pd.concat(reviews, ignore_index=True) 
    reviews.sort_values(by='date', inplace=True)
    firstDateAvailable = pd.to_datetime('2021-03-01')
    reviews = reviews[reviews['date'] > firstDateAvailable]
    os.system("mkdir -p ./data")
    reviews.to_json('./data/yelp_academic_dataset_review.json', orient='records', lines=True)
    valid_users = reviews['user_id'].unique()
    users = []  # List to store chunks
    for chunk in pd.read_json(datapath / "yelp_academic_dataset_user.json", lines=True, chunksize = chunk_size):
        users.append(chunk)
    users = pd.concat(users, ignore_index=True) 
    filtered_users = users[users['user_id'].isin(valid_users)]
    filtered_users.to_json('./data/yelp_academic_dataset_user.json', orient='records', lines=True)
    businesses = []  # List to store chunks
    for chunk in pd.read_json(datapath / "yelp_academic_dataset_business.json", lines=True, chunksize = chunk_size):
        businesses.append(chunk)
    businesses = pd.concat(businesses, ignore_index=True) 
    valid_businesses = reviews['business_id'].unique()
    filtered_business = businesses[businesses['business_id'].isin(valid_businesses)]
    filtered_business.to_json('./data/yelp_academic_dataset_business.json', orient='records', lines=True)
    photos_json_path = './raw-data/Yelp-Photos/Yelp Photos/yelp_photos/photos.json'
    photos_json = pd.read_json(photos_json_path, lines=True)
    filtered_json = photos_json[photos_json['business_id'].isin(valid_businesses)]
    filtered_json.to_json('./data/filtered_photos.json', orient='records', lines=True)
    valid_photos = filtered_json['photo_id'].unique()
    source_folder = './raw-data/Yelp-Photos/Yelp Photos/yelp_photos/photos'
    source_paths = [os.path.join(source_folder, photo_id) for photo_id in valid_photos if os.path.exists(os.path.join(source_folder, f"{photo_id}.jpg"))]
    print(source_paths)
    destination_folder = './data/photos/'
    os.system(f"mkdir -p {destination_folder}")
    if source_paths:
        command = ['cp'] + source_paths + [destination_folder]
        str_command = " ".join(command)
        os.system(str_command)
        print(f"Copied {len(source_paths)} files.")

if __name__ == "__main__":
    condense_data()