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

Store the data in the root folder, after downloading from [here](https://duke.box.com/s/00jahivpjl2m9fl2nqnh3hhtrxzjdwhg).

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


### Deployment

#### VCM

If you're deploying on vcm, change the vcm base url in `frontend/.env.production`. You can change the link to the url / ip address of the server you are hosting it on, if using GCP or Azure for deployment.

Commands to deploy:

```bash
cd TemplateProject # You can rename this, just make sure the current directory has the docker compose file
sudo docker compose -f docker-compose.yml up --build -d
```