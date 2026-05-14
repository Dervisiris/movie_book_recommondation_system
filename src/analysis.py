import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

class Analyzer:

    def __init__(self,ratings:pd.DataFrame,items:pd.DataFrame,output_dir:str):
        self.ratings =ratings
        self.items = items
        self.output_dir = output_dir
        os.makedirs(output_dir,exist_ok=True)
    
    def basic_stats(self):
        merged = self.ratings.merge(self.items,on="item_id",how="left")
        return {
            "number_of_users" : int(self.ratings["user_id"].nunique()),
            "number_of_items" : int(self.items["item_id"].nunique()),
            "total_score" : int(len(self.ratings)),
            "avg_score" : float(round(self.ratings["rating"].mean(),2)),
            "most_active_user": int(
                self.ratings["user_id"].value_counts().idxmax()
            ),
            "most_popular_item" : str(
                merged.groupby("title").size().sort_values(ascending=False).index[0]
            ),
            "most_popular_category" : str(
                merged.groupby("category").size().sort_values(ascending=False).index[0]
            ),
        }
    

    def rating_distribution_chart(self):
        path = os.path.join(self.output_dir,"rating_distribution.png")
        plt.figure(figsize=(7,4))
        counts = self.ratings["rating"].value_counts().sort_index()
        plt.bar(counts.index.astype(str), counts.values, color= "#4C9AFF")
        plt.title("Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Registration Order")
        plt.tight_layout()
        plt.savefig(path,dpi= 120)
        plt.close()
        return path
    
    def category_popularity_chart(self):
        path = os.path.join(self.output_dir, "category_popularity.png")
        merged = self.ratings.merge(self.items,on="item_id",how="left")
        cat_counts = merged.groupby("category").size().sort_values(ascending=True)
        plt.figure(figsize=(7,4))
        plt.barh(cat_counts.index, cat_counts.values, color = "#36B37E")
        plt.title("Number of Points by Category ")
        plt.xlabel("Number of Point")
        plt.tight_layout()
        plt.savefig(path,dpi=120)
        plt.close
        return path
    
    def top_rated_items_chart(self,top_n: int = 10):
        path = os.path.join(self.output_dir,"top_rated_items.png")
        merged = self.ratings.merge(self.items, on="item_id",how="left")
        agg = merged.groupby("title").agg(
            average = ("rating","mean"),
            number = ("rating","count")
        )
        agg = agg[agg["number"] >= 2].sort_values("average",ascending=True).tail(top_n)
        plt.figure(figsize=(8,5))
        plt.barh(agg.index, agg["average"], color= "#FF8B00")
        plt.title(f"The {top_n} Item with the Highest Average Rating")
        plt.xlabel("Average Point")
        plt.xlim(0,5)
        plt.tight_layout()
        plt.savefig(path,dpi=120)
        plt.close()
        return path
    
    




        