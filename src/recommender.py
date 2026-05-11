import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    
    def __init__(self, ratings: pd.DataFrame, items: pd.DataFrame):
        
        self.ratings = ratings
        self.items = items
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
    
    def build_user_item_matrix(self, fill_value :float = 0.0):

        self.user_item_matrix = self.ratings.pivot_table(
            index="user_id",
            columns="item_id",
            fill_value=fill_value,
        )
        return self.user_item_matrix
    
    def compute_user_similarity(self):
        if self.user_item_matrix is None:
            self.build_user_item_matrix()
        sim = cosine_similarity(self.user_item_matrix.values)
        self.user_similarity = pd.DataFrame(
            sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )
        return self.user_similarity
    
    def compute_item_similarity(self):
        if self.user_item_matrix is None:
            self.build_user_item_matrix()
        
        sim = cosine_similarity(self.user_item_matrix.T.values)
        self.item_similarity = pd.DataFrame(
            sim,
            index=self.user_item_matrix.columns
            columns=self.user_item_matrix.columns,
        )
        return self.item_similarity
    
    def recommend_user_based(self,user_id:int, top_n:int = 5, k_neighbors: int = 3):
        if self.user_similarity is None:
            self.compute_user_similarity()
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"user_id: {user_id} is not found in data set.")
        
        sim_scores =self.user_similarity[user_id].drop(user_id)
        top_neighbors = sim_scores.sort_values(ascending=False).head(k_neighbors)

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = set(user_ratings[user_ratings > 0].index)

        neighbor_ratings = self.user_item_matrix.loc[top_neighbors.index]

        ### Gonna continue tomorrow...
