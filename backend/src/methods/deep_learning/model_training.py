import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class YelpDataset(Dataset):
    def __init__(self, user_ids, business_ids, ratings, user_feats, business_feats):
        self.user_ids = user_ids
        self.business_ids = business_ids
        self.ratings = ratings
        self.user_feats = user_feats
        self.business_feats = business_feats

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.business_ids[idx],
                self.user_feats[idx], self.business_feats[idx],
                self.ratings[idx])

# Model class
class DeepRecommender(nn.Module):
    def __init__(self, num_users, num_businesses, user_feat_dim, biz_feat_dim, emb_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.business_embedding = nn.Embedding(num_businesses, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2 + user_feat_dim + biz_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, business_ids, user_feats, business_feats):
        u_emb = self.user_embedding(user_ids)
        b_emb = self.business_embedding(business_ids)
        x = torch.cat([u_emb, b_emb, user_feats, business_feats], dim=1)
        return self.fc(x)

    @torch.no_grad()
    def recommend_for_user(self, user_id, user_feat, business_ids, business_feats, top_k=10):
        self.eval()
        user_ids_tensor = torch.tensor([user_id] * len(business_ids), dtype=torch.long)
        user_feats_tensor = torch.tensor([user_feat] * len(business_ids), dtype=torch.float32)
        business_ids_tensor = torch.tensor(business_ids, dtype=torch.long)
        business_feats_tensor = torch.tensor(business_feats, dtype=torch.float32)

        predictions = self.forward(user_ids_tensor, business_ids_tensor, user_feats_tensor, business_feats_tensor).squeeze()
        top_indices = torch.topk(predictions, top_k).indices
        return [(int(business_ids[i]), float(predictions[i])) for i in top_indices]


def business_data_preprocessing(business_df):

    # only look at restaurants
    def check_categories(categories):
        """Return all food related categories"""
        try:
            return 'Restaurants' in categories.split(', ')
        except Exception:
            return False

    business_df = business_df[business_df['categories'].apply(check_categories)]

    # rename stars to rating
    business_df.rename(columns={'stars': 'rating', 'review_count': 'business_review_count', 'name': 'business_name'}, inplace=True)

    # only keep open businesses
    business_df = business_df[business_df['is_open'] == 1]

    return business_df


def user_data_preprocessing(user_df):
    # rename review_count to user_review_count
    user_df.rename(columns={'review_count': 'user_review_count', 'name': 'user_name'}, inplace=True)
    return user_df

# Data loading and training
if __name__ == '__main__':
    print("Starting data preprocessing...")
    user_df = pd.read_json('data/yelp_academic_dataset_user.json', lines=True)
    user_df = user_data_preprocessing(user_df)
    print("User data preprocessing complete.")

    business_df = pd.read_json('data/yelp_academic_dataset_business.json', lines=True)
    business_df = business_data_preprocessing(business_df)
    print("Business data preprocessing complete.")

    review_df = pd.read_json('data/yelp_academic_dataset_review.json', lines=True)

    merged = review_df.merge(user_df, on='user_id').merge(business_df, on='business_id')

    # Encode user and business ids using LabelEncoder   
    le_user = LabelEncoder()
    le_business = LabelEncoder()
    merged['user_id_enc'] = le_user.fit_transform(merged['user_id'])
    merged['business_id_enc'] = le_business.fit_transform(merged['business_id'])
    print("Label encoding complete.")

    os.makedirs('backend/models', exist_ok=True)

    # Save the LabelEncoder objects
    joblib.dump(le_user, 'backend/models/le_user.pkl')
    joblib.dump(le_business, 'backend/models/le_business.pkl')
    print("LabelEncoder objects saved.")

    user_features = merged[['user_review_count', 'average_stars']].fillna(0).values.astype(np.float32)
    business_features = merged[['business_review_count', 'rating']].fillna(0).values.astype(np.float32)
    print("User and business features extracted.")

    # Scale user and business features
    scaler = StandardScaler()
    user_features = scaler.fit_transform(user_features)
    business_features = scaler.fit_transform(business_features)
    print("User and business features scaled.")

    X_user_ids = torch.tensor(merged['user_id_enc'].values, dtype=torch.long).to(device)
    X_business_ids = torch.tensor(merged['business_id_enc'].values, dtype=torch.long).to(device)
    y_ratings = torch.tensor(merged['stars'].values, dtype=torch.float32).to(device)
    user_feats_tensor = torch.tensor(user_features, dtype=torch.float32).to(device)
    biz_feats_tensor = torch.tensor(business_features, dtype=torch.float32).to(device)
    print("User and business features converted to tensors and moved to device.")

    X_train, X_test, u_train, u_test, b_train, b_test, y_train, y_test = train_test_split(
        list(zip(X_user_ids, X_business_ids)), user_feats_tensor, biz_feats_tensor, y_ratings,
        test_size=0.2, random_state=42
    )
    print("Train-test split complete.")

    train_user_ids = torch.tensor([u for u, _ in X_train], dtype=torch.long).to(device)
    train_business_ids = torch.tensor([b for _, b in X_train], dtype=torch.long).to(device)
    print("Train user and business ids converted to tensors and moved to device.")

    train_dataset = YelpDataset(train_user_ids, train_business_ids, y_train, u_train, b_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Train dataset created and dataloader initialized.")

    model = DeepRecommender(
        num_users=merged['user_id_enc'].nunique(),
        num_businesses=merged['business_id_enc'].nunique(),
        user_feat_dim=user_features.shape[1],
        biz_feat_dim=business_features.shape[1]
    )
    model.to(device)
    print("Model initialized.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Criterion and optimizer initialized.")

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for user_ids, business_ids, user_feats, biz_feats, ratings in train_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, business_ids, user_feats, biz_feats).squeeze()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'backend/models/deep_recommender.pth')
