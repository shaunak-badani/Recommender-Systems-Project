import numpy as np
from typing import List, Dict, Union, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RecommendationEvaluator:
    def __init__(self):
        pass

    def calculate_rating_metrics(self,
                               true_ratings: np.ndarray,
                               predicted_ratings: np.ndarray) -> Dict[str, float]:
        """
        Calculate MAE and RMSE for rating predictions.
        """
        mae = mean_absolute_error(true_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))

        return {
            'mae': mae,
            'rmse': rmse
        }

    def calculate_precision_recall(self,
                                 true_items: List[int],
                                 recommended_items: List[int],
                                 k: int = 10) -> Dict[str, float]:
        """
        Calculate precision and recall at k.
        """
        recommended_items_at_k = recommended_items[:k]

        true_set = set(true_items)
        rec_set = set(recommended_items_at_k)

        true_positives = len(true_set.intersection(rec_set))

        precision = true_positives / len(recommended_items_at_k) if recommended_items_at_k else 0
        recall = true_positives / len(true_items) if true_items else 0

        return {
            f'precision@{k}': precision,
            f'recall@{k}': recall
        }

    def calculate_mrr(self,
                     true_items: List[int],
                     recommended_items: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        """
        for i, item in enumerate(recommended_items, 1):
            if item in true_items:
                return 1.0 / i
        return 0.0

    def calculate_ndcg(self,
                      true_items: List[int],
                      recommended_items: List[int],
                      k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        """
        recommended_items_at_k = recommended_items[:k]

        dcg = 0
        for i, item in enumerate(recommended_items_at_k, 1):
            if item in true_items:
                # Relevance is binary (1 if relevant, 0 otherwise)
                dcg += 1 / np.log2(i + 1)

        # Ideal DCG calculation
        idcg = 0
        ideal_k = min(len(true_items), k)
        for i in range(1, ideal_k + 1):
            idcg += 1 / np.log2(i + 1)

        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg

    def evaluate_recommendations(self,
                               true_ratings: np.ndarray = None,
                               predicted_ratings: np.ndarray = None,
                               true_items: List[int] = None,
                               recommended_items: List[int] = None,
                               k: int = 10) -> Dict[str, float]:
        """
        Comprehensive evaluation function.
        Calculates relevant metrics based on provided inputs.
        """
        results = {}

        # Rating prediction metrics
        if true_ratings is not None and predicted_ratings is not None:
            rating_metrics = self.calculate_rating_metrics(true_ratings, predicted_ratings)
            results.update(rating_metrics)

        # Ranking metrics
        if true_items is not None and recommended_items is not None:
            precision_recall = self.calculate_precision_recall(true_items, recommended_items, k)
            results.update(precision_recall)

            mrr = self.calculate_mrr(true_items, recommended_items)
            results['mrr'] = mrr

            ndcg = self.calculate_ndcg(true_items, recommended_items, k)
            results[f'ndcg@{k}'] = ndcg

        return results 