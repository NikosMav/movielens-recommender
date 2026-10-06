"""Classic collaborative-filtering baselines."""

from movielens_recommender.baselines.als import ALSRecommender
from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.baselines.rp3beta import RP3betaRecommender

__all__ = [
    "MostPopularRecommender",
    "ItemItemCosineRecommender",
    "ALSRecommender",
    "EASERecommender",
    "RP3betaRecommender",
]
