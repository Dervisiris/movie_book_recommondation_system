import pandas as pd
import os



class DataLoader:
    
    def __init__(self,data_dir : str):
        
        self.data_dir = data_dir
        self.ratings_path = os.path.join(data_dir,"ratings.csv")
        self.items_path = os.path.join(data_dir,"items.csv")
        self.ratings = None
        self.items = None

    
    def load(self):
        if not os.path.exists(self.ratings_path):
            raise FileNotFoundError(f"ratings.cvs file could not found. {self.ratings_path}")
        if not os.path.exists(self.items_path):
            raise FileNotFoundError(f"items.csv file could not found. {self.items_path}")
        
        self.ratings = pd.read_csv(self.ratings_path)
        self.items = pd.read_csv(self.items_path)
        
        return self.items, self.ratings
    

    def clean(self, min_rating: int = 1, max_rating: int = 5):
        
        if self.ratings is None or self.items is None:
            raise ValueError("Call load() function first.")
        
        before = len(self.ratings)
        self.ratings = self.ratings.dropna(subset = ["user_id","item_id","rating"])


        self.ratings = (
            self.ratings.groupby(["user_id","item_id"],as_index =False)
            .agg({"rating": "mean","timestamp":"max"}) 
        )

        after = len(self.ratings)
        print(f"[DataLoader] cleaning: {before} --> {after} row ")
        return self.ratings, self.items
    
    def merge(self):
        if self.ratings is None or self.items is None:
            raise ValueError("Call load() function first.")
        return self.ratings.merge(self.items, on="item_id", how="left")
    

    def summary(self):
        if self.ratings is None or self.items is None:
            raise ValueError("Call load() function first.")
        
        return {
            "number_of_users" : int(self.ratings["user_id"].nunique()),
            "number_of_items" : int(self.items["item_id"].nunique()),
            "score_record" : int(len(self.ratings)),
            "average_score" : float(round(self.ratings["rating"].mean(), 2)),
            "point density" : round(
                len(self.ratings) / (self.ratings["user_id"].nunique() * self.items["item_id"].nunique()), 3,
            ),
                                    
        }


        

