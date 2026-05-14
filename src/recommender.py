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
            values="rating",
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
            index=self.user_item_matrix.columns,
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

        weighted_sum = neighbor_ratings.T.dot(top_neighbors)
        sim_sum = top_neighbors.sum()

        if sim_sum == 0:
            scores = pd.Series(0,index=neighbor_ratings.columns)
        else:
            scores = weighted_sum / sim_sum
        
        scores = scores.drop(labels=[i for i in rated_items if i in scores.index])

        scores = scores[scores > 0].sort_values(ascending=False).head(top_n)

        return self._format_recommendations(scores, method="user_based")
    
    
    def recommend_item_based(self, user_id: int, top_n: int = 5):
        if self.item_similarity is None:
            self.compute_item_similarity()
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"user_id:{user_id} could not found in data set.")
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]

        if rated_items.empty:
            return pd.DataFrame(columns=["item_id","title","category","type","score"])
        
        sim_matrix = self.item_similarity
        scores = sim_matrix[rated_items.index].dot(rated_items) / (sim_matrix[rated_items.index].abs().sum(axis=1) + 1e-9)

        scores = scores.drop(labels=rated_items.index,errors="ignore")
        scores =scores[scores > 0].sort_values(ascending=False).head(top_n)
        
        return self._format_recommendations(scores, method= "item_based")
    
    def _format_recommendations(self, scores: pd.Series,method:str):
        if scores.empty:
            return pd.DataFrame(columns=["item_id","title","category","type","score","method"])
        df = scores.reset_index()
        df.columns = ["item_id","score"]
        df = df.merge(self.items,on="item_id",how="left")
        df["score"] = df["score"].round(3)
        df["method"] = method

        return df[["item_id","title","category","type","year","score","method"]]
    

