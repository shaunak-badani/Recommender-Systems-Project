import os
import sys
import torch
import pandas as pd
import numpy as np  
import joblib

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)
from scripts.model_training import DeepRecommender, business_data_preprocessing, user_data_preprocessing

# Load data
user_df = pd.read_json('../../data/yelp_academic_dataset_user.json', lines=True)
user_df = user_data_preprocessing(user_df)

business_df = pd.read_json('../../data/yelp_academic_dataset_business.json', lines=True)
business_df = business_data_preprocessing(business_df, inference=True)

# Load encoders and scalers
le_user = joblib.load('../models/le_user.pkl')
le_business = joblib.load('../models/le_business.pkl')
user_scaler = joblib.load('../models/user_scaler.pkl')
business_scaler = joblib.load('../models/business_scaler.pkl')

# Filter users to only those that were in training data
known_users = set(le_user.classes_)
user_df = user_df[user_df['user_id'].isin(known_users)]

# Filter businesses to only those that were in training data
known_businesses = set(le_business.classes_)
business_df = business_df[business_df['business_id'].isin(known_businesses)]

# Transform IDs using the loaded encoders
user_df['user_id_enc'] = le_user.transform(user_df['user_id'])
business_df['business_id_enc'] = le_business.transform(business_df['business_id'])

# Setup encoders
def get_user_enc_map():
    return dict(zip(user_df['user_id'], user_df['user_id_enc']))

def get_business_enc_map():
    return dict(zip(business_df['business_id'], business_df['business_id_enc']))

def get_business_dec_map():
    return dict(zip(business_df['business_id_enc'], business_df['business_name']))

user_enc_map = get_user_enc_map()
business_enc_map = get_business_enc_map()
business_dec_map = get_business_dec_map()

# Prepare model
# Read configuration to ensure we're using the right dimensions
with open('../models/model_config.txt', 'r') as f:
    lines = f.readlines()
    user_feat_dim = int(lines[0].split(':')[1].strip())
    biz_feat_dim = int(lines[1].split(':')[1].strip())

model = DeepRecommender(
    num_users=len(le_user.classes_),
    num_businesses=len(le_business.classes_),
    user_feat_dim=user_feat_dim,
    biz_feat_dim=biz_feat_dim
)
model.load_state_dict(torch.load('../models/deep_recommender.pth'))
model.eval()

# Inference
def recommend_for_user(user_id_str, top_k=10, verbose=True):
    if user_id_str not in user_enc_map:
        raise ValueError(f"User ID '{user_id_str}' not found in training data. Cannot make recommendations for new users.")

    user_id = user_enc_map[user_id_str]
    to_remove = ['user_id', 'user_name', 'user_id_enc']
    user_feats = [feat for feat in user_df.columns.tolist() if feat not in to_remove]
    
    # Get the actual user's feature values
    user_row = user_df[user_df['user_id'] == user_id_str].iloc[0]
    user_feat = user_row[user_feats].values.astype(np.float32).reshape(1, -1)
    
    # Apply the same scaling as during training
    user_feat = user_scaler.transform(user_feat)[0]

    # Filter to only include businesses in cities where the user has reviewed before
    user_reviews = pd.read_json('../../data/yelp_academic_dataset_review.json', lines=True)
    user_reviews = user_reviews[user_reviews['user_id'] == user_id_str]
    
    if len(user_reviews) > 0:
        reviewed_business_ids = user_reviews['business_id'].unique()
        reviewed_businesses = business_df[business_df['business_id'].isin(reviewed_business_ids)]
        
        if len(reviewed_businesses) > 0:
            visited_cities = set(reviewed_businesses['city'])
            candidate_businesses = business_df[business_df['city'].isin(visited_cities)]
        else:
            candidate_businesses = business_df
    else:
        # If no reviews, use all businesses
        candidate_businesses = business_df
    
    # Filter out businesses the user has already reviewed
    if len(user_reviews) > 0:
        candidate_businesses = candidate_businesses[~candidate_businesses['business_id'].isin(reviewed_business_ids)]
    
    if verbose:
        print(f"\nCreating recommendations for user with {len(user_reviews)} previous reviews")
        print(f"Found {len(candidate_businesses)} candidate businesses")
    
    if len(candidate_businesses) == 0:
        return []

    candidate_biz_ids = candidate_businesses['business_id_enc'].values
    
    # Extract all business features using the same columns as in training
    to_remove = ['business_id', 'business_name', 'business_id_enc', 'city', 'state']
    business_feats = [feat for feat in business_df.columns.tolist() if feat not in to_remove]
    biz_feats = candidate_businesses[business_feats].fillna(0).values.astype(np.float32)
    
    # Apply the same scaling as during training
    biz_feats = business_scaler.transform(biz_feats)

    recommendations = model.recommend_for_user(
        user_id, user_feat, candidate_biz_ids, biz_feats, top_k=top_k
    )

    results = []
    for biz_id, score in recommendations:
        biz_name = business_dec_map.get(biz_id, "Unknown")
        biz_info = candidate_businesses[candidate_businesses['business_id_enc'] == biz_id].iloc[0]
        
        results.append({
            'name': biz_name,
            'rating_predicted_for_user': score,
            'city': biz_info['city'],
            'state': biz_info['state'],
            'restaurant_rating': biz_info['rating']
        })


    # convert to actual business ids
    business_ids = [le_business.inverse_transform([reco[0]])[0] for reco in recommendations]
    print(business_ids)
    return results, business_ids

def display_recommendations(recommendations):
    if not recommendations:
        print("No recommendations found.")
        return
        
    print("\nTop Recommended Restaurants:")
    print("=" * 80)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} - {rec['city']}, {rec['state']}")
        print(f"   Predicted Rating: {rec['rating_predicted_for_user']:.2f} (Restaurant Rating: {rec['restaurant_rating']:.1f})")
        print("-" * 80)

if __name__ == '__main__':
    user_input = input("Enter user_id: ").strip()
    recs, _ = recommend_for_user(user_input, top_k=10)
    display_recommendations(recs)