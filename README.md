# Template project

> A template project with React & Vite as frontend and fastapi as backend.


### How to run

- Frontend
```
cd frontend
npm install
npm run dev
```
- The application is then hosted on `localhost:5173`


- Backend
1. Store the data in the root folder, after downloading from [here](https://duke.app.box.com/s/00jahivpjl2m9fl2nqnh3hhtrxzjdwhg)
2. Store the data in the root folder, after downloading from [here](https://duke.box.com/s/4of0k1j2ymfirv908rczov0jtj92nop8).

The application expects the following tree:

```
.
├── LICENSE
├── README.md
├── backend
├── data
├── docker-compose-localhost.yml
├── docker-compose-vcm.yml
├── frontend
├── notebooks
└── scripts
```
And within data the following structure:
```
data
├── filtered_photos.json
├── not-available.png
├── photos
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
└── yelp_academic_dataset_user.json
```

Running the backend:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd backend/src
uvicorn main:app --reload
```


### Recommendation Methods

#### Naive Approach (`/mean` endpoint)

This method provides baseline recommendations. It works as follows:
1.  **Identify User Locations:** Finds all the unique cities and states where the user has previously reviewed businesses using the `yelp_academic_dataset_review.json` and `yelp_academic_dataset_business.json` files.
2.  **Find Top-Rated Businesses:** Filters the `yelp_academic_dataset_business.json` data to find businesses located in those identified cities/states.
3.  **Rank and Filter:** Ranks these candidate businesses based on their star rating (descending) and review count (descending). It also applies a minimum review count filter (e.g., businesses must have at least 5 reviews).
4.  **Exclude Reviewed:** Removes businesses from the list that the user has already reviewed.
5.  **Return Top N:** Returns the IDs of the top N remaining businesses as recommendations.

#### Deep Learning Approach (`/deep-learning` endpoint)

This method uses a neural collaborative filtering. It works as follows:

1. **Data Preprocessing:**
   - Filters businesses to only include restaurants
   - Processes user and business features including review counts, ratings, engagement scores, and more
   - Encodes user and business IDs using LabelEncoder
   - Scales features using StandardScaler

2. **Model Architecture:**
   - Uses a neural network with embedding layers for users and businesses
   - Combines embeddings with additional user and business features
   - Implements batch normalization and dropout for better training
   - Outputs ratings constrained to 1-5 range using sigmoid activation

3. **Training Process:**
   - Uses MSE loss and Adam optimizer
   - Implements early stopping to prevent overfitting
   - Saves the best model based on validation loss
   - Stores necessary encoders and scalers for inference

4. **Inference:**
   - Filters recommendations to businesses in cities where the user has previously reviewed
   - Excludes businesses the user has already reviewed
   - Returns top N recommendations with predicted ratings and business details

5. **Features Used:**
   - User Features: user_review_count, yelping_years, reviews_per_year, engagement_score, engagement_per_review, total_compliments, compliments_per_review, compliment_type_count, num_friends, friend_density, friend_to_fan_ratio, elite_years_count, is_elite, elite_score, tough_reviewer, social_compliments, thoughtful_compliments
   - Business Features: business_review_count, rating, popularity, high_rating, curated attributes, top N categories one-hot encoded, weekly_hours, open_weekends, open_late, days_open


### Deployment

#### VCM

If you're deploying on vcm, change the vcm base url in `frontend/.env.production`. You can change the link to the url / ip address of the server you are hosting it on, if using GCP or Azure for deployment.

Commands to deploy:

```bash
cd TemplateProject # You can rename this, just make sure the current directory has the docker compose file
sudo docker compose -f docker-compose.yml up --build -d
```