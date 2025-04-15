from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from methods.traditional.collaborative_filtering import CollaborativeRecommender
from utils import Utils, load_and_prep_data
from fastapi.responses import JSONResponse
import time
from methods.naive.top_rated_by_location import NaiveRecommender
from methods.deep_learning.inference import recommend_for_user
from utils import Utils
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path

app = FastAPI(root_path='/api')

# Allowed origins for CORS
origins = [
    "http://localhost:5173",
    "http://vcm-45508.vm.duke.edu"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals: Load data and initialize recommender at startup --- #
print("API Startup: Loading data and initializing recommender...")
start_load_time = time.time()
data_path = Path('../../data') # Relative to main.py (in src/)
collaborative_recommender_instance = None
try:
    # Use *all* ratings for the API instance
    all_ratings_df, _, restaurant_embeddings = load_and_prep_data(data_path)
    collaborative_recommender_instance = CollaborativeRecommender(all_ratings_df, restaurant_embeddings)
    print(f"Data loaded and recommender initialized in {time.time() - start_load_time:.2f} seconds.")
except Exception as e:
    print(f"FATAL: API failed to load data or initialize recommender: {e}")
    # Endpoints will return an error if recommender is None
# --- End Globals --- #

@app.get("/")
async def root():
    return {"message": "Hello world!"}

# Placeholder - Not implemented
@app.get("/mean")
def get_naive_recommendations(user_id: str):
    """
    Get user recommendations using the naive approach (top-rated in user's cities).
    Also fetches the user's name.
    """

    user_name = user_id # Default to user_id if name not found
    try:
        datapath = Path("../../data")
        user_file = datapath / "yelp_academic_dataset_user.json"
        chunk_size = 10000
        user_df = None
        if user_file.exists():
            for chunk in pd.read_json(user_file, lines=True, chunksize=chunk_size):
                user_chunk = chunk[chunk['user_id'] == user_id]
                if not user_chunk.empty:
                    user_name = user_chunk.iloc[0]['name']
                    break # Found the user, no need to read more chunks
        else:
            print(f"Warning: User data file not found at {user_file}")
    except Exception as e:
        print(f"Error loading or searching user data: {e}")
        # Keep user_name as user_id

    # --- Generate Recommendations ---
    recommender = NaiveRecommender()
    recommender.load_data()
    # Pass the minimum review count filter if desired, e.g., min_review_count=5
    recommended_ids = recommender.generate_recommendations(user_id, min_review_count=5)

    if not recommended_ids:
         # Return user name even if no recommendations found
         return JSONResponse(content={"user_name": user_name, "recommendations": []}, status_code=200) # Use 200 OK as it's not strictly an error if user exists but has no recs

    # Wrap the data using the existing utility function
    recommended_restaurant_data = Utils.gift_wrap_restaurant_data(recommended_ids)
    
    # Return user name and recommendations
    return JSONResponse(content={"user_name": user_name, "recommendations": recommended_restaurant_data})


@app.get("/traditional")
def get_user_recommendations(user_id: str):
    """Get item-item collaborative filtering recommendations for a user."""
    if collaborative_recommender_instance is None:
         return JSONResponse(status_code=500, content={"error": "Recommender not initialized due to data loading issues."}) 

    recommended_ids = collaborative_recommender_instance.generate_recommendations(user_id)

    if not recommended_ids:
        # Check if user exists in the loaded ratings
        if user_id not in collaborative_recommender_instance.ratings['user_id'].unique():
             return JSONResponse(status_code=404, content={"error": "User not found or has no ratings."}) 
        else:
             # User exists, but couldn't generate recommendations
             return JSONResponse(status_code=404, content={"error": "Could not generate recommendations for this user (e.g., data sparsity, missing embeddings)."}) 

    top_recommended_restaurant_data = Utils.gift_wrap_restaurant_data(recommended_ids)
    return JSONResponse(content=top_recommended_restaurant_data)

# Placeholder - Not implemented
@app.get("/deep-learning")
def query_deep_learning_model(user_id: str):
    """
    Query endpoint for the deep learning model
    """
    _, business_ids = recommend_for_user(user_id)

    user_name = user_id
    try:
        datapath = Path("../../data")
        user_file = datapath / "yelp_academic_dataset_user.json"
        chunk_size = 10000
      
        if user_file.exists():
            for chunk in pd.read_json(user_file, lines=True, chunksize=chunk_size):
                user_chunk = chunk[chunk['user_id'] == user_id]
                if not user_chunk.empty:
                    user_name = user_chunk.iloc[0]['name']
                    break # Found the user, no need to read more chunks
        else:
            print(f"Warning: User data file not found at {user_file}")
    except Exception as e:
        print(f"Error loading or searching user data: {e}")

    recommended_restaurant_data = Utils.gift_wrap_restaurant_data(business_ids)
    return JSONResponse(content={"user_name": user_name, "recommendations": recommended_restaurant_data})
