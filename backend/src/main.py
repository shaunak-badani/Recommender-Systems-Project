from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from methods.traditional.collaborative_filtering import CollaborativeRecommender

app = FastAPI(root_path='/api')

# list of allowed origins
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

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.get("/mean")
def query_mean_model(query: str):
    """
    Query endpoint for the mean model
    """
    # Pass query to some function
    answer = f"Response to the mean query : {query}"
    # answer = f(query) 
    return {"answer": answer}

@app.get("/traditional")
def get_user_recommendations(user_id: str):
    """
    Get user recommendations using collaborative filtering
    """
    answer = f"Response to the traditional query : {user_id}"
    recommender = CollaborativeRecommender()
    recommender.load_data()
    top_rated_restaurant, most_similar_restaurants_ids = recommender.generate_recommendations(user_id)
    print(most_similar_restaurants_ids)
    if not top_rated_restaurant:
        return {"error": "User does not have past reviews!"}
    return {"answer": top_rated_restaurant}


@app.get("/deep-learning")
def query_deep_learning_model(query: str):
    """
    Query endpoint for the deep learning model
    """
    # Pass query to some function
    answer = f"Response to the deep learning model query : {query}"
    # answer = f(query) 
    return {"answer": answer}
