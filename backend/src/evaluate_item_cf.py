import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time

from eval import RecommendationEvaluator
from methods.traditional.collaborative_filtering import CollaborativeRecommender
from utils import load_and_prep_data, PopularityRecommender

# --- Evaluation Configuration ---
k_recommendations = 10
relevance_threshold = 3.0  # Lowered from 4.0
test_split_size = 0.2
split_random_state = 42
DATA_SUBSET_FRACTION = 0.20
# --- End Configuration ---

def evaluate_item_collaborative_filtering(k=k_recommendations, relevance_threshold=relevance_threshold,
                                      test_size=test_split_size, random_state=split_random_state,
                                      subset_fraction=DATA_SUBSET_FRACTION):
    """Evaluates the item-item collaborative filtering approach."""
    start_time = time.time()
    print(f"Starting Item-Item CF evaluation (k={k}, threshold={relevance_threshold})...")
    datapath = Path('../../data') 

    # 1. Load Data & Build Embeddings
    try:
        ratings_df_full, business_df, restaurant_embeddings = load_and_prep_data(datapath)
        if ratings_df_full.empty or restaurant_embeddings.empty:
            print("Error: Data loading resulted in empty dataframes.")
            return
    except Exception as e:
        print(f"Failed during data load or embedding creation: {e}")
        return

    # Subsetting Data (Optional)
    if subset_fraction is not None and 0 < subset_fraction < 1.0:
        print(f"\n--- Using a {subset_fraction:.0%} subset ---")
        all_user_ids = ratings_df_full['user_id'].unique()
        subset_user_ids = np.random.choice(all_user_ids, 
                                          size=int(len(all_user_ids) * subset_fraction), 
                                          replace=False)
        ratings_df = ratings_df_full[ratings_df_full['user_id'].isin(subset_user_ids)]
        print(f"Subset size: {len(ratings_df)} reviews from {len(subset_user_ids)} users.")
    else:
        print("\n--- Using the full dataset ---")
        ratings_df = ratings_df_full

    # 2. Train/Test Split - Time-based if date column exists
    print(f"\nSplitting ratings into train/test ({1-test_size:.0%}/{test_size:.0%})...")
    
    if 'date' in ratings_df.columns:
        # Time-based split
        print("Using time-based split...")
        ratings_df['date'] = pd.to_datetime(ratings_df['date'])
        ratings_df = ratings_df.sort_values(by='date')
        split_idx = int(len(ratings_df) * (1 - test_size))
        train_ratings_df = ratings_df.iloc[:split_idx]
        test_ratings_df = ratings_df.iloc[split_idx:]
    else:
        # Random split as fallback
        print("Date column not found, using random split...")
        try:
            train_ratings_df, test_ratings_df = train_test_split(
                ratings_df, test_size=test_size, random_state=random_state, 
                stratify=ratings_df['user_id']
            )
        except ValueError as e:
            print(f"Warning: Stratify failed ({e}). Splitting without stratify.")
            train_ratings_df, test_ratings_df = train_test_split(
                ratings_df, test_size=test_size, random_state=random_state
            )
    
    print(f"Train ratings: {len(train_ratings_df)}, Test ratings: {len(test_ratings_df)}")

    # 3. Instantiate Recommenders with TRAINING data only
    print("Instantiating recommenders...")
    
    # Popularity baseline
    print("Setting up popularity baseline recommender...")
    popularity_recommender = PopularityRecommender(train_ratings_df)
    
    # Collaborative Filtering recommender
    print("Setting up collaborative filtering recommender...")
    try:
        cf_recommender = CollaborativeRecommender(
            ratings_df=train_ratings_df,
            restaurant_embeddings_df=restaurant_embeddings
        )
    except Exception as e:
        print(f"Error instantiating CollaborativeRecommender: {e}")
        return

    # 4. Prepare for Evaluation
    evaluator = RecommendationEvaluator()
    
    # Store metrics for each recommender
    all_metrics = {
        'popularity': defaultdict(list),
        'collaborative': defaultdict(list)
    }
    
    users_evaluated = {
        'popularity': 0,
        'collaborative': 0
    }

    print("Evaluating users found in the test set...")
    # Find relevant test items (user's ratings >= threshold)
    test_user_relevant_items = test_ratings_df[test_ratings_df['stars'] >= relevance_threshold].groupby('user_id')['business_id'].apply(list)
    test_users = test_user_relevant_items.index
    total_test_users_with_relevant = len(test_users)
    print(f"Found {total_test_users_with_relevant} users with relevant items in test set.")

    # 5. Generate recommendations and evaluate for each test user
    for i, user_id in enumerate(test_users):
        true_items = test_user_relevant_items[user_id]
        
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
        
        # Evaluate collaborative filtering
        cf_recommendations = cf_recommender.generate_recommendations(user_id, k=k)
        if cf_recommendations:
            cf_metrics = evaluator.evaluate_recommendations(
                true_items=true_items,
                recommended_items=cf_recommendations,
                k=k
            )
            for metric_name, value in cf_metrics.items():
                all_metrics['collaborative'][metric_name].append(value)
            users_evaluated['collaborative'] += 1

        if (i + 1) % 1000 == 0:
            print(f"...processed {i + 1}/{total_test_users_with_relevant} users.")

    print(f"Finished evaluating users (Popularity: {users_evaluated['popularity']}, " 
          f"Collaborative: {users_evaluated['collaborative']})")

    # 6. Aggregate and Print Metrics
    print("\n--- Recommendation Evaluation Results ---")
    print(f"Dataset: {'Full' if subset_fraction is None or subset_fraction >= 1.0 else f'{subset_fraction:.0%}'}")
    print(f"Train Reviews: {len(train_ratings_df)}, Test Reviews: {len(test_ratings_df)}")
    print(f"Users w/ Relevant Test Items: {total_test_users_with_relevant}")
    
    for recommender_name in ['popularity', 'collaborative']:
        users_count = users_evaluated[recommender_name]
        metrics = all_metrics[recommender_name]
        
        print(f"\n{recommender_name.upper()} RECOMMENDER:")
        print(f"Users Evaluated: {users_count}")
        
        if users_count > 0:
            user_coverage = users_count / total_test_users_with_relevant
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
    evaluate_item_collaborative_filtering()