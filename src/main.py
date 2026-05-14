import os 
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR,"src"))

from data_loader import DataLoader
from recommender import Recommender
from analysis import Analyzer

DATA_DIR = os.path.join(BASE_DIR,"data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

def banner():
    print("-" * 60)
    print("Movie / Book Recommondation System ")
    print("Cosine Similarity-Based Recommendation Engine")
    print("-" * 60)



def show_summary(loader:DataLoader,analyzer:Analyzer):
    summary = loader.summary()
    stats = analyzer.basic_stats()
    print("\n ---Data Sets Summary---")

    for k,v in summary.items():
        print(f" {k:25s}: {v}")

    print(f"  {'most_active_user':25s}: {stats['most_active_user']}")
    print(f"  {'most_popular_item':25s}: {stats['most_popular_item']}")
    print(f"  {'most_popular_category':25s}: {stats['most_popular_category']}")


def show_recommendations(rec: Recommender, user_id:int,top_n:int = 5):
    print(f"\n>>> User-Based Recommendations for user {user_id}")
    ub = rec.recommend_user_based(user_id=user_id,top_n=top_n,k_neighbors=3)
    print(ub.to_string(index=False) if not ub.empty else "(No suggestion was generated)")

    print(f">>> Item-Based Recommendations for user {user_id}")
    ib = rec.recommend_item_based(user_id=user_id,top_n=top_n)
    print(ib.to_string(index=False) if not ib.empty else "(No suggestion was generated)")

    return ub,ib

def export_recommendations(ub:pd.DataFrame,ib:pd.DataFrame,user_id:int):
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR,"recommendations.csv")
    combined = pd.concat([ub,ib],ignore_index=True)
    combined.insert(0,"user_id",user_id)
    combined.to_csv(out_path,index=False,encoding="UTF-8")
    print(f"\n[OK] The suggestions have been recorded in the file: {out_path}")


def generate_charts(analyzer:Analyzer):
    print("\n---Charts are being Generated---")
    p1 = analyzer.rating_distribution_chart()
    p2 = analyzer.category_popularity_chart()
    p3 = analyzer.top_rated_items_chart(top_n=10)
    for p in (p1,p2,p3):
        print(f"[OK] {p}")


def interactive_menu(loader,rec,analyzer):

    while True:
        print("\n" + "-" * 40)
        print(" 1) Show data set summary")
        print(" 2) Generate a recommendation for a user")
        print(" 3) Generate charts")
        print(" 4) Exit")
        choice = input("Choice: ").strip()

        if choice == "1":
            show_summary(loader,analyzer)
        elif choice == "2":
            try:
                uid =int(input("Enter user ID (example: 1-15):").strip())
                ub, ib = show_recommendations(rec,uid,top_n=5) 
                save = input("Should I save the results as a CSV file? (Y/N):").strip().lower()
                if save == "Y":
                    export_recommendations(ub,ib,uid)
            except ValueError as e: 
                print(f"Error : {e}")
        elif choice == "3":
            generate_charts(analyzer)
        elif choice == "4":
            print("See you soon ! ")
            break
        
        else:
            print("Unvalid choice. :))")

def run_demo(loader,rec,analyzer):
    show_summary(loader,analyzer)
    demo_user = 1 
    ub, ib = show_recommendations(rec, demo_user,top_n=5)
    export_recommendations(ub,ib,demo_user)
    generate_charts(analyzer)



def main():
    banner()


    loader = DataLoader(DATA_DIR)
    loader.load()
    rating, items = loader.clean()


    rec = Recommender(rating,items)
    rec.build_user_item_matrix(fill_value=0.0)
    rec.compute_user_similarity()
    rec.compute_item_similarity()


    analyzer = Analyzer(rating,items, CHARTS_DIR)


    if "--demo" in sys.argv:
        run_demo(loader,rec,analyzer)
    else:
        interactive_menu(loader,rec,analyzer)
    


if __name__ == "__main__":
        main()












