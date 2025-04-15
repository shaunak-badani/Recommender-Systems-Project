import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

from eval import RecommendationEvaluator
from utils import PopularityRecommender

# Config
k_recommendations = 10
relevance_threshold = 3.0
test_split_size = 0.2
split_random_state = 42
USER_SUBSET_FRACTION = 0.05

def evaluate_popularity_recommender(k=k_recommendations, relevance_threshold=relevance_threshold,
                          test_size=test_split_size, random_state=split_random_state,
                          subset_fraction=USER_SUBSET_FRACTION):
    start_time = time.time()
    print(f"Starting evaluation (k={k}, threshold={relevance_threshold})...")
    datapath = Path('../../data')

    # Loading data
    print("Loading review data...")
    reviews_df_full = pd.read_json(datapath / "yelp_academic_dataset_review.json", lines=True)
    ratings_df = reviews_df_full[['user_id', 'business_id', 'stars', 'date']].copy()
    del reviews_df_full
    ratings_df.dropna(subset=['user_id', 'business_id', 'stars'], inplace=True)
    print(f"Loaded {len(ratings_df)} reviews.")

    # Filtering for restaurants
    print("Loading business data...")
    businesses_df = pd.read_json(datapath / "yelp_academic_dataset_business.json", lines=True)
    print(f"Loaded {len(businesses_df)} businesses.")
    
    restaurants_mask = businesses_df['categories'].str.contains('Restaurant', case=False, na=False)
    restaurant_businesses = businesses_df[restaurants_mask]
    print(f"Filtered to {len(restaurant_businesses)} restaurant businesses.")
    
    restaurant_ids = set(restaurant_businesses['business_id'])
    ratings_df = ratings_df[ratings_df['business_id'].isin(restaurant_ids)]
    print(f"Filtered to {len(ratings_df)} restaurant reviews.")

    # Subsetting users
    all_user_ids = ratings_df['user_id'].unique()
    print(f"Total users with restaurant reviews: {len(all_user_ids)}")

    if subset_fraction < 1.0:
        print(f"Sampling {subset_fraction:.0%} of users...")
        np.random.seed(random_state)
        users_to_evaluate = np.random.choice(
            all_user_ids, 
            size=int(len(all_user_ids) * subset_fraction), 
            replace=False
        )
        print(f"Evaluating {len(users_to_evaluate)} users.")
        evaluation_ratings_df = ratings_df[ratings_df['user_id'].isin(users_to_evaluate)].copy()
    else:
        users_to_evaluate = all_user_ids
        evaluation_ratings_df = ratings_df.copy()

    # Doing splitting
    print(f"Splitting into train/test...")
    evaluation_ratings_df['date'] = pd.to_datetime(evaluation_ratings_df['date'])
    evaluation_ratings_df = evaluation_ratings_df.sort_values(by='date')
    
    split_idx = int(len(evaluation_ratings_df) * (1 - test_size))
    train_ratings_df = evaluation_ratings_df.iloc[:split_idx]
    test_ratings_df = evaluation_ratings_df.iloc[split_idx:]
    
    print(f"Train: {len(train_ratings_df)}, Test: {len(test_ratings_df)}")

    # Setting up metrics
    evaluator = RecommendationEvaluator()
    metrics = defaultdict(list)
    users_evaluated = 0
    
    # Creating recommender
    print("Setting up recommender...")
    popularity_recommender = PopularityRecommender(train_ratings_df)

    # Finding test set ground truth
    print("Finding test items...")
    test_user_relevant_items = test_ratings_df[test_ratings_df['stars'] >= relevance_threshold].groupby('user_id')['business_id'].apply(list)
    test_users = test_user_relevant_items.index
    total_test_users = len(test_users)
    print(f"Found {total_test_users} users with relevant test items.")

    # Running evaluation
    print("Evaluating...")
    for i, user_id in enumerate(test_users):
        true_items = test_user_relevant_items[user_id]
        if not true_items:
            continue

        # Get popularity recommendations
        recommendations = popularity_recommender.generate_recommendations(user_id, k=k)
        if recommendations:
            user_metrics = evaluator.evaluate_recommendations(
                true_items=true_items,
                recommended_items=recommendations,
                k=k
            )
            for metric_name, value in user_metrics.items():
                metrics[metric_name].append(value)
            users_evaluated += 1
            
        if (i + 1) % 100 == 0:
            print(f"...processed {i + 1}/{total_test_users} users")

    # Printing results
    print(f"\nFinished evaluation: {users_evaluated} users")

    print("\n--- Results ---")
    print(f"Dataset: {subset_fraction:.0%}")
    print(f"Restaurant reviews: {len(ratings_df)}")
    print(f"Train: {len(train_ratings_df)}, Test: {len(test_ratings_df)}")
    print(f"Users with relevant items: {total_test_users}")
    
    print(f"\nPOPULARITY RECOMMENDER:")
    print(f"Users evaluated: {users_evaluated}")
    
    if users_evaluated > 0:
        user_coverage = users_evaluated / total_test_users
        print(f"Coverage: {user_coverage:.2%}")
        
        print(f"Metrics (k={k}, threshold={relevance_threshold}):")
        for metric_name in sorted(metrics.keys()):
            avg_value = np.mean(metrics[metric_name])
            readable_name = metric_name.replace(f'@{k}', f'@{k}').replace('mrr', 'MRR').replace('ndcg', 'NDCG').replace('precision', 'Precision').replace('recall', 'Recall')
            padding = " " * (12 - len(readable_name))
            print(f"  {readable_name}:{padding}{avg_value:.4f}")
    else:
        print("No users evaluated.")

    print(f"\nTime: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    print("Starting evaluation...")
    evaluate_popularity_recommender()