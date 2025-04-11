import torch
import pandas as pd
import numpy as np
from model_training import DeepRecommender, business_data_preprocessing, user_data_preprocessing    
import joblib

# Load data
user_df = pd.read_json('data/yelp_academic_dataset_user.json', lines=True)
user_df = user_data_preprocessing(user_df)

business_df = pd.read_json('data/yelp_academic_dataset_business.json', lines=True)
business_df = business_data_preprocessing(business_df)

# Load encoders
le_user = joblib.load('backend/models/le_user.pkl')
le_business = joblib.load('backend/models/le_business.pkl')

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
user_feat_dim = 2
biz_feat_dim = 2
model = DeepRecommender(
    num_users=len(le_user.classes_),  # Use the number of classes from the encoder
    num_businesses=len(le_business.classes_),  # Use the number of classes from the encoder
    user_feat_dim=user_feat_dim,
    biz_feat_dim=biz_feat_dim
)
model.load_state_dict(torch.load('backend/models/deep_recommender.pth'))
model.eval()

# Inference

def recommend_for_user(user_id_str, top_k=10):
    if user_id_str not in user_enc_map:
        raise ValueError(f"User ID '{user_id_str}' not found in training data. Cannot make recommendations for new users.")

    user_id = user_enc_map[user_id_str]
    user_row = user_df[user_df['user_id'] == user_id_str].iloc[0]
    user_feat = np.array([user_row['user_review_count'], user_row['average_stars']], dtype=np.float32)

    candidate_biz_ids = business_df['business_id_enc'].values
    biz_feats = business_df[['business_review_count', 'rating']].fillna(0).values.astype(np.float32)

    recommendations = model.recommend_for_user(
        user_id, user_feat, candidate_biz_ids, biz_feats, top_k=top_k
    )

    decoded = [(business_dec_map[biz_id], score) for biz_id, score in recommendations]
    return decoded

if __name__ == '__main__':
    user_input = input("Enter user_id: ").strip()
    try:
        recs = recommend_for_user(user_input, top_k=10)
        print("\nTop Recommended Restaurants:")
        for name, score in recs:
            print(f"{name} - Predicted Rating: {score:.2f}")
    except Exception as e:
        print(f"Error: {e}")
