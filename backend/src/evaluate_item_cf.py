import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time

from eval import RecommendationEvaluator
from methods.traditional.collaborative_filtering import CollaborativeRecommender
from utils import load_and_prep_data

# --- Evaluation Configuration ---
k_recommendations = 10
relevance_threshold = 4.0
test_split_size = 0.2
split_random_state = 42
# Use only a fraction of data for quick testing. Set to 1.0 or None to use full dataset.
DATA_SUBSET_FRACTION = 0.20
# --- End Configuration ---

def evaluate_item_collaborative_filtering(k=k_recommendations, relevance_threshold=relevance_threshold,
                                      test_size=test_split_size, random_state=split_random_state,
                                      subset_fraction=DATA_SUBSET_FRACTION):
    """Evaluates the item-item collaborative filtering approach.
    Uses a train/test split and calculates ranking metrics.
    """
    start_time = time.time()
    print(f"Starting Item-Item CF evaluation (k={k}, threshold={relevance_threshold})...")
    datapath = Path('../../data') 

    # 1. Load Data & Build Embeddings
    try:
        ratings_df_full, _, restaurant_embeddings = load_and_prep_data(datapath)
        if ratings_df_full.empty or restaurant_embeddings.empty:
             print("Error: Data loading resulted in empty dataframes.")
             return
    except Exception as e:
        print(f"Failed during data load or embedding creation: {e}")
        return

    # Subsetting Data (Optional)
    if subset_fraction is not None and 0 < subset_fraction < 1.0:
        print(f"\n--- Using a {subset_fraction:.0%} subset ({len(ratings_df_full) * subset_fraction:,.0f} reviews) ---")
        # Sample users first to keep user integrity if possible, then get their ratings
        # This is slightly more complex than just sampling ratings, but better represents user behavior
        all_user_ids = ratings_df_full['user_id'].unique()
        subset_user_ids = np.random.choice(all_user_ids, size=int(len(all_user_ids) * subset_fraction), replace=False)
        ratings_df = ratings_df_full[ratings_df_full['user_id'].isin(subset_user_ids)]
        # Alternatively, simpler sampling:
        # ratings_df = ratings_df_full.sample(frac=subset_fraction, random_state=random_state)
        print(f"Subset size: {len(ratings_df)} reviews from {len(subset_user_ids)} users.")
    else:
        print("\n--- Using the full dataset ---")
        ratings_df = ratings_df_full

    # 2. Train/Test Split
    print(f"\nSplitting ratings into train/test ({1-test_size:.0%}/{test_size:.0%})...")
    try:
        train_ratings_df, test_ratings_df = train_test_split(
            ratings_df, test_size=test_size, random_state=random_state, stratify=ratings_df['user_id']
        )
    except ValueError as e:
        # Handles cases where stratification might fail (e.g., users with only 1 rating)
        print(f"Warning: Stratify failed ({e}). Splitting without stratify.")
        train_ratings_df, test_ratings_df = train_test_split(
            ratings_df, test_size=test_size, random_state=random_state
        )
    print(f"Train ratings: {len(train_ratings_df)}, Test ratings: {len(test_ratings_df)}")

    # 3. Instantiate Recommender with TRAINING data
    print("Instantiating CollaborativeRecommender...")
    try:
        recommender = CollaborativeRecommender(
            ratings_df=train_ratings_df,
            restaurant_embeddings_df=restaurant_embeddings
        )
    except Exception as e:
        print(f"Error instantiating CollaborativeRecommender: {e}")
        return

    # 4. Prepare for Evaluation
    evaluator = RecommendationEvaluator()
    metrics_accumulator = defaultdict(list)
    users_evaluated = 0

    print("Evaluating users found in the test set...")
    # Determine ground truth relevant items for users in the test set
    test_user_relevant_items = test_ratings_df[test_ratings_df['stars'] >= relevance_threshold].groupby('user_id')['business_id'].apply(list)
    test_users = test_user_relevant_items.index
    total_test_users_with_relevant = len(test_users)
    print(f"Found {total_test_users_with_relevant} users with relevant items (rating >= {relevance_threshold}) in test set.")

    # 5. Generate recommendations and evaluate for each test user
    for i, user_id in enumerate(test_users):
        true_items = test_user_relevant_items[user_id]
        recommended_items = recommender.generate_recommendations(user_id, k=k)

        if not recommended_items:
             continue # Skip users if no recommendations generated

        # Calculate metrics for this user
        user_metrics = evaluator.evaluate_recommendations(
            true_items=true_items,
            recommended_items=recommended_items,
            k=k
        )

        # Accumulate metrics
        for metric_name, value in user_metrics.items():
             metrics_accumulator[metric_name].append(value)
        users_evaluated += 1

        if (i + 1) % 1000 == 0:
            print(f"...processed {i + 1}/{total_test_users_with_relevant} users.")

    print(f"Finished evaluating {users_evaluated} users (recommendations generated)." )

    # 6. Aggregate and Print Metrics
    print("\n--- Item-Item Collaborative Filtering Evaluation Results ---")
    print(f"Dataset Used: {'Full' if subset_fraction is None or subset_fraction >= 1.0 else f'{subset_fraction:.0%}'} ({len(ratings_df)} reviews)")
    print(f"Train Reviews: {len(train_ratings_df)}, Test Reviews: {len(test_ratings_df)}")
    print(f"Users w/ Relevant Test Items: {total_test_users_with_relevant}")
    print(f"Users Evaluated (Recs Generated): {users_evaluated}")

    if users_evaluated > 0:
        user_coverage = users_evaluated / total_test_users_with_relevant if total_test_users_with_relevant > 0 else 0
        print(f"User Coverage (Evaluated / Total w/ Relevant): {user_coverage:.2%}")

        print(f"\nRanking Metrics (k={k}, relevance_threshold={relevance_threshold}):")
        for metric_name in sorted(metrics_accumulator.keys()):
             avg_value = np.mean(metrics_accumulator[metric_name])
             readable_name = metric_name.replace(f'@{k}', f'@{k}').replace('mrr', 'MRR').replace('ndcg', 'NDCG').replace('precision', 'Precision').replace('recall', 'Recall')
             padding = " " * (12 - len(readable_name))
             print(f"  {readable_name}:{padding}{avg_value:.4f}")
    else:
        print("\nNo users were successfully evaluated. Cannot calculate average metrics.")

    print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds.")
    print("-------------------------------------------------")

if __name__ == "__main__":
    evaluate_item_collaborative_filtering() 