# Learning Guide

This is a small notebook where I write down the things I learned while
building this project. I keep it simple, just for myself, so I can come back
and remember how things work.

---

## 1. Project Setup

I started with this folder structure:

```
recommendation_system/
├── data/
├── src/
├── outputs/
├── requirements.txt
└── README.md
```

I wrote the code in four separate files inside `src/`:
- `data_loader.py`
- `recommender.py`
- `analysis.py`
- `main.py`

The order I wrote them in was important. I learned that I should write the
small parts first (the ones that do not depend on others), and then write
`main.py` at the end, because `main.py` uses everything else.

---

## 2. What Each File Does

### data_loader.py
Reads the CSV files and cleans them. I removed rows with missing values and
made sure that all ratings are between 1 and 5. If the same user rated the
same item twice, I take the average.

### recommender.py
This was the hardest part. It does three things:
1. Builds a user-item matrix.
2. Calculates how similar users are to each other (and how similar items
   are to each other).
3. Returns recommendations.

### analysis.py
Creates statistics and charts. I used `matplotlib` for the charts.

### main.py
Brings everything together and has an interactive menu.

---

## 3. Important Things I Learned

### Why we use `fill_value=0` in the pivot table

When I build the user-item matrix, many cells are empty because most users
have not rated most items. If I leave them as `NaN`, the cosine similarity
function does not work. So I fill them with `0`.

`0` means "this user has not rated this item". It does not mean
"this user gave 0 stars". It is just a neutral value that allows the math
to work.

### The transpose trick

For **user similarity**, I do not use transpose:
```python
cosine_similarity(self.user_item_matrix.values)
```

For **item similarity**, I use transpose:
```python
cosine_similarity(self.user_item_matrix.T.values)
```

This is because `cosine_similarity` compares the **rows** of a matrix.
In my pivot table, users are in rows and items are in columns. So:
- To compare users, the matrix is already in the right shape.
- To compare items, I need to put items in rows, which means I need to
  transpose the matrix.

### The weighted average formula

For user-based recommendations, the score for an item is:

```
score = sum(similarity * rating) / sum(similarity)
```

I divide by `sum(similarity)` to bring the score back to the 1-5 range.
Without that, the numbers would be too big and would not make sense.

This is just like a weighted average. The neighbors that are more similar
to me have more weight in the final score.

### Why I drop already-rated items

After calculating scores, I drop the items that the user already rated.
If I do not do this, the system will recommend items the user already
saw. That is useless. Netflix and Spotify also do this; they never
recommend a movie you already watched in the "recommended" list.

### `if __name__ == "__main__":`

This line tells Python: "Only run this code when this file is run
directly. Do not run it if another file imports this one."

It is important because later, when I write a FastAPI service, the
service will import my modules. I do not want random code to run when
that happens.

---

## 4. Errors I Made and How I Fixed Them

I had several bugs in my first version. Writing them down so I do not
forget:

1. I wrote `if __name__ == "__main__":` **inside** the `main()` function.
   That meant `main()` was never called. I had to move that block to the
   bottom of the file, with no indentation.

2. In `pivot_table`, I forgot to add `values="rating"`. Without it, pandas
   tried to pivot all numeric columns including `timestamp`, which broke
   the matrix.

3. In `data_loader.py`, I returned `self.items, self.ratings` instead of
   `self.ratings, self.items`. The order was wrong and inconsistent with
   `clean()`. I learned that consistency between methods matters.

4. I wrote `plt.close` without parentheses in one place. That is just a
   reference to the function, not a function call. The figure stayed in
   memory.

5. I asked the user "(Y/N):" but my code was checking for "e" (Turkish
   "evet"). Small mistake but the save feature did not work.

---

## 5. Things I Want to Improve Later

- Add unit tests with `pytest`
- Try Pearson correlation instead of cosine similarity
- Handle the "cold start" problem (new users with no ratings)
- Try matrix factorization (SVD) for missing value prediction

---

## 6. Next Step

Turn this project into an API with **FastAPI** and build a user interface
with **Flutter** that works on both web and mobile. The Python part is
ready, now I need to connect it to a real application.