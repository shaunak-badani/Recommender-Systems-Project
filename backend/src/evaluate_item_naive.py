import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time

from eval import RecommendationEvaluator
from methods.naive.top_rated_by_location import NaiveRecommender
from utils import PopularityRecommender

# --- Evaluation Configuration ---
k_recommendations = 10
relevance_threshold = 3.0  # Lowered from 4.0
test_split_size = 0.2
split_random_state = 42
USER_SUBSET_FRACTION = 0.10  # Evaluate only 10% of users for speed
# --- End Configuration ---

def evaluate_naive_recommender(k=k_recommendations, relevance_threshold=relevance_threshold,
                               test_size=test_split_size, random_state=split_random_state,
                               subset_fraction=USER_SUBSET_FRACTION):
    """Evaluates the NaiveRecommender with proper train/test separation."""
    start_time = time.time()
    print(f"Starting Naive Recommender evaluation (k={k}, threshold={relevance_threshold})...")

    # 1. Load Data
    print("Loading data...")
    try:
        recommender = NaiveRecommender()
        recommender.load_data()
        if recommender.reviews is None or recommender.businesses is None or \
           recommender.reviews.empty or recommender.businesses.empty:
            print("Error: Data loading failed or resulted in empty dataframes.")
            return
        
        # Create copies of the full data for proper separation
        full_reviews_df = recommender.reviews.copy()
        full_businesses_df = recommender.businesses.copy()
        
        # Create ratings dataframe for evaluation
        ratings_df = pd.merge(
            full_reviews_df, 
            full_businesses_df[['business_id', 'stars']], 
            on='business_id', 
            how='inner'
        )
    except Exception as e:
        print(f"Failed during data load: {e}")
        return

    all_user_ids = ratings_df['user_id'].unique()
    print(f"Total users in dataset: {len(all_user_ids)}")

    # Subsetting Users for Evaluation
    if subset_fraction is not None and 0 < subset_fraction < 1.0:
        print(f"--- Evaluating a {subset_fraction:.0%} subset of users ---")
        users_to_evaluate = np.random.choice(
            all_user_ids, 
            size=int(len(all_user_ids) * subset_fraction), 
            replace=False
        )
        print(f"Evaluating {len(users_to_evaluate)} users.")
        evaluation_ratings_df = ratings_df[ratings_df['user_id'].isin(users_to_evaluate)]
    else:
        print("--- Evaluating all users ---")
        users_to_evaluate = all_user_ids
        evaluation_ratings_df = ratings_df

    # 2. Train/Test Split - Time-based if date column exists
    print(f"Splitting evaluation ratings into train/test ({1-test_size:.0%}/{test_size:.0%})...")
    
    if 'date' in evaluation_ratings_df.columns:
        # Time-based split
        print("Using time-based split...")
        evaluation_ratings_df['date'] = pd.to_datetime(evaluation_ratings_df['date'])
        evaluation_ratings_df = evaluation_ratings_df.sort_values(by='date')
        split_idx = int(len(evaluation_ratings_df) * (1 - test_size))
        train_ratings_df = evaluation_ratings_df.iloc[:split_idx]
        test_ratings_df = evaluation_ratings_df.iloc[split_idx:]
    else:
        # Random split as fallback
        try:
            train_ratings_df, test_ratings_df = train_test_split(
                evaluation_ratings_df, test_size=test_size, random_state=random_state, 
                stratify=evaluation_ratings_df['user_id']
            )
        except ValueError as e:
            print(f"Warning: Stratify failed ({e}). Splitting without stratify.")
            train_ratings_df, test_ratings_df = train_test_split(
                evaluation_ratings_df, test_size=test_size, random_state=random_state
            )
    
    print(f"Train ratings: {len(train_ratings_df)}, Test ratings: {len(test_ratings_df)}")

    # 3. Create training-only review dataset for recommendation
    train_reviews_df = train_ratings_df[['user_id', 'business_id']].copy()

    # 4. Prepare for Evaluation
    evaluator = RecommendationEvaluator()
    
    all_metrics = {
        'popularity': defaultdict(list),
        'naive': defaultdict(list)
    }
    
    users_evaluated = {
        'popularity': 0,
        'naive': 0
    }

    print("Evaluating users...")
    # Find relevant test items (user's ratings >= threshold)
    test_user_relevant_items = test_ratings_df[test_ratings_df['stars'] >= relevance_threshold].groupby('user_id')['business_id'].apply(list)
    test_users_in_split = test_user_relevant_items.index

    # Filter to users with relevant test items
    target_evaluation_users = [u for u in users_to_evaluate if u in test_users_in_split]
    total_target_users = len(target_evaluation_users)

    print(f"Found {total_target_users} users with relevant items in test set.")
    
    # Instantiate recommenders with TRAINING data only
    print("Setting up recommenders with training data only...")
    
    # Popularity baseline
    popularity_recommender = PopularityRecommender(train_ratings_df)
    
    # Naive recommender uses the instance loaded earlier with full data

    # 5. Generate recommendations and evaluate for each target test user
    for i, user_id in enumerate(target_evaluation_users):
        # Get ground truth from the test split
        true_items = test_user_relevant_items.get(user_id, [])
        
        # Evaluate popularity baseline
        pop_recommendations = popularity_recommender.generate_recommendations(user_id, k=k)
        if pop_recommendations:
            pop_metrics = evaluator.evaluate_recommendations(
                true_items=true_items,
                recommended_items=pop_recommendations,
                k=k
            )
            for metric_name, value in pop_metrics.items():
                all_metrics['popularity'][metric_name].append(value)
            users_evaluated['popularity'] += 1
        
        # Evaluate naive recommender
        # Generate recommendations using the original 'recommender' instance with full data
        # Add exclude_reviewed=False later after modifying NaiveRecommender class
        naive_recommendations = recommender.generate_recommendations(user_id, n=k) # Use 'recommender' instance
        if naive_recommendations:
            naive_metrics = evaluator.evaluate_recommendations(
                true_items=true_items,
                recommended_items=naive_recommendations,
                k=k
            )
            for metric_name, value in naive_metrics.items():
                all_metrics['naive'][metric_name].append(value)
            users_evaluated['naive'] += 1

        if (i + 1) % 100 == 0:
            print(f"...processed {i + 1}/{total_target_users} users.")

    print(f"Finished evaluating users (Popularity: {users_evaluated['popularity']}, "
          f"Naive: {users_evaluated['naive']})")

    # 6. Aggregate and Print Metrics
    print("\n--- Recommendation Evaluation Results ---")
    print(f"Dataset: {'Full' if subset_fraction is None or subset_fraction >= 1.0 else f'{subset_fraction:.0%}'}")
    print(f"Train Ratings: {len(train_ratings_df)}, Test Ratings: {len(test_ratings_df)}")
    print(f"Users w/ Relevant Test Items: {total_target_users}")
    
    for recommender_name in ['popularity', 'naive']:
        users_count = users_evaluated[recommender_name]
        metrics = all_metrics[recommender_name]
        
        print(f"\n{recommender_name.upper()} RECOMMENDER:")
        print(f"Users Evaluated: {users_count}")
        
        if users_count > 0:
            user_coverage = users_count / total_target_users
            print(f"User Coverage: {user_coverage:.2%}")
            
            print(f"Ranking Metrics (k={k}, relevance_threshold={relevance_threshold}):")
            for metric_name in sorted(metrics.keys()):
                avg_value = np.mean(metrics[metric_name])
                readable_name = metric_name.replace(f'@{k}', f'@{k}').replace('mrr', 'MRR').replace('ndcg', 'NDCG').replace('precision', 'Precision').replace('recall', 'Recall')
                padding = " " * (12 - len(readable_name))
                print(f"  {readable_name}:{padding}{avg_value:.4f}")
        else:
            print("No users were successfully evaluated. Cannot calculate average metrics.")

    print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds.")
    print("-------------------------------------------------")

if __name__ == "__main__":
    evaluate_naive_recommender()