# Movie / Book Recommendation System

This is my graduation project for the Python AI 201 course.
It is a simple recommendation engine that suggests movies and books to users
based on their past ratings.

I built two types of recommendations:
- **User-based**: find similar users, then suggest what they liked.
- **Item-based**: find similar items to the ones the user already liked.

Both methods use **cosine similarity** to measure how close two users
(or two items) are to each other.

---

## Folder Structure

```
recommendation_system/
├── data/
│   ├── ratings.csv
│   └── items.csv
├── src/
│   ├── data_loader.py
│   ├── recommender.py
│   ├── analysis.py
│   └── main.py
├── outputs/
│   └── charts/
├── requirements.txt
└── README.md
```

- `data/` keeps the CSV files.
- `src/` keeps the Python code.
- `outputs/` is where the program writes results and charts.

---

## How to Run

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the program in demo mode (quick test):
   ```bash
   python src/main.py --demo
   ```

3. Or run it in interactive menu mode:
   ```bash
   python src/main.py
   ```

In the menu you can:
- See a summary of the data set
- Get recommendations for a user
- Generate charts about the data

---

## Modules

### data_loader.py
Reads `ratings.csv` and `items.csv`. Removes empty rows and keeps only ratings
between 1 and 5. If the same user rated the same item twice, it takes the
average.

### recommender.py
This is the main part of the project.
- Builds a user-item matrix using pandas `pivot_table`.
- Computes user similarity and item similarity with `cosine_similarity`
  from scikit-learn.
- Returns the top N recommendations for a given user.

### analysis.py
Creates simple statistics and three charts:
- Rating distribution
- Category popularity
- Top rated items

### main.py
Connects all the modules together and gives an interactive menu.

---

## Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib

---

## What I Learned

- How to clean data with pandas
- How `pivot_table` works
- What cosine similarity is and how to compute it
- The difference between user-based and item-based recommendations
- How to split code into different files (modular structure)
- Why `if __name__ == "__main__":` is important

---

## Next Steps

The next step is to turn this project into a real application using
**FastAPI** for the backend and **Flutter** for the user interface. I will keep updating this repository.