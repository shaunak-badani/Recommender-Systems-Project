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
from scripts.data_preprocessing import user_data_preprocessing, business_data_preprocessing

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
    def __init__(self, num_users, num_businesses, user_feat_dim, biz_feat_dim, emb_dim=32):  # Reduced embedding dimension
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.business_embedding = nn.Embedding(num_businesses, emb_dim)
        
        # Add batch normalization layers and reduce network complexity
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2 + user_feat_dim + biz_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # Initialize weights to help with convergence
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, business_ids, user_feats, business_feats):
        u_emb = self.user_embedding(user_ids)
        b_emb = self.business_embedding(business_ids)
        x = torch.cat([u_emb, b_emb, user_feats, business_feats], dim=1)
        raw_output = self.fc(x)
        # Use a constrained output range (1-5) for ratings
        return 1.0 + 4.0 * torch.sigmoid(raw_output)

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

# Data loading and training
if __name__ == '__main__':
    print("Starting data preprocessing...")
    user_df = pd.read_json('data/yelp_academic_dataset_user.json', lines=True)
    user_df = user_data_preprocessing(user_df)

    to_remove = ['user_id', 'user_name']
    user_feats = [feat for feat in user_df.columns.tolist() if feat not in to_remove]
    print("User data preprocessing complete. User features: ", user_feats)

    business_df = pd.read_json('data/yelp_academic_dataset_business.json', lines=True)
    business_df = business_data_preprocessing(business_df)
    to_remove = ['business_id', 'business_name']
    business_feats = [feat for feat in business_df.columns.tolist() if feat not in to_remove]
    print("Business data preprocessing complete. Business features: ", business_feats)

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

    # Enhanced user features
    user_features = merged[user_feats].fillna(0).values.astype(np.float32)
    
    # Enhanced business features
    business_features = merged[business_feats].fillna(0).values.astype(np.float32)
    
    print("user and business features extracted.")

    # Scale user and business features
    user_scaler = StandardScaler()
    business_scaler = StandardScaler()
    
    user_features = user_scaler.fit_transform(user_features)
    business_features = business_scaler.fit_transform(business_features)
    
    # Save the scalers for inference
    joblib.dump(user_scaler, 'backend/models/user_scaler.pkl')
    joblib.dump(business_scaler, 'backend/models/business_scaler.pkl')
    print("User and business features scaled and scalers saved.")

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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Increased batch size
    
    # Create validation dataset
    val_user_ids = torch.tensor([u for u, _ in X_test], dtype=torch.long).to(device)
    val_business_ids = torch.tensor([b for _, b in X_test], dtype=torch.long).to(device)
    val_dataset = YelpDataset(val_user_ids, val_business_ids, y_test, u_test, b_test)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)  # Increased batch size
    print("Train and validation datasets created and dataloaders initialized.")

    model = DeepRecommender(
        num_users=merged['user_id_enc'].nunique(),
        num_businesses=merged['business_id_enc'].nunique(),
        user_feat_dim=user_features.shape[1],
        biz_feat_dim=business_features.shape[1]
    )
    model.to(device)
    print("Model initialized.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # Added learning rate scheduler
    print("Criterion, optimizer, and scheduler initialized.")

    # Add early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(20):  # Increased epochs
        # Training
        model.train()
        epoch_loss = 0
        for user_ids, business_ids, user_feats, biz_feats, ratings in train_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, business_ids, user_feats, biz_feats).squeeze()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_ids, business_ids, user_feats, biz_feats, ratings in val_loader:
                outputs = model(user_ids, business_ids, user_feats, biz_feats).squeeze()
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'backend/models/deep_recommender.pth')
            print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("Training complete!")
    
    # Save model architecture details for reference
    with open('backend/models/model_config.txt', 'w') as f:
        f.write(f"User features: {user_features.shape[1]}\n")
        f.write(f"Business features: {business_features.shape[1]}\n")