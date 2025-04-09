from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from methods.traditional.collaborative_filtering import CollaborativeRecommender
from utils import Utils, load_and_prep_data
from fastapi.responses import JSONResponse
from pathlib import Path
import time

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
data_path = Path('../data') # Relative to main.py (in src/)
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
def query_mean_model(query: str):
    answer = f"Response to the mean query : {query}"
    return {"answer": answer}

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
def query_deep_learning_model(query: str):
    answer = f"Response to the deep learning model query : {query}"
    return {"answer": answer}
