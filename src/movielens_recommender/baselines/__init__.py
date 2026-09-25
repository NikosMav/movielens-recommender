"""Classic collaborative-filtering baselines."""

from movielens_recommender.baselines.als import ALSRecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.baselines.popular import MostPopularRecommender

__all__ = [
    "MostPopularRecommender",
    "ItemItemCosineRecommender",
    "ALSRecommender",
]
