Collaborative Filtering Movie Recommender System
================================================


This project implements collaborative filtering recommendation system in memory-based approaches and model based approaches, specifically, user-based collaborative filtering, item-based collaborative filtering, clustering-based approach (k nearest neighbors), matrix factorization approach (nonnegative matrix factorization), alternating gradient descent, and neuron network approach, with data taken from MovieLens dataset, to predict user ratings for movies and recommend movies to users based on the prediction. Root mean square error (RMSE) and area under the curve (AUC) are used as the evaluation metrics.

Database website: https://grouplens.org/datasets/movielens/

## Project structure:

Movie-Recommender directory: Contains python code that implement various algorithms to recommend movie.

SQL directory: Contains SQL statements that extract, aggregate and merge the data and the output tables.

dataOverview directory: Contains plots and tables that are generated about the dataset features.

dataset: Contains MovieLens dataset.

doc directory: Contains project report and poster.

reference directory: Contains usefull papers and tutorials.

small_dataset.db: Built in a local database software, in order to using sql to analyze the dataset.

```bash
C:.
│   LICENSE
│   README.md
│   small_dataset.db
│
├───dataOverview
│       dataset_analysis.py
│       Distribution of movie numbers among rating variance.png
│       Distribution of Ratings among Movies.png
│       Distribution of Ratings among Users.png
│       Frequency of Rating Values.png
│       frequent_appeared_genres_of_high_rated_movies.csv
│       frequent_appeared_genres_of_high_rated_movies.py
│       high_relevance_tags_of_high_rated_movies.csv
│       high_relevance_tags_of_high_rated_movies.py
│       movieId_sum(rating)_mean.py
│
├───dataset
│       links.csv
│       movies.csv
│       ratings.csv
│       README.txt
│       tags.csv
│
├───doc
│       Poster.pdf
│       Report.pdf
│
├───Movie-Recommender
│       knn.py
│       memory_based_cf.py
│       nmf.py
│       nn.py
│
├───reference
│       1205.3193.pdf
│       9783319296579-c1.pdf
│       collab-filtering-tutorial.ppt
│       nmf_nature.pdf
│
└───SQL
        genres-count(movieId).csv
        genres-count(movieId).sql
        high_rate_movie_high_relevant_tage_count(movieId).csv
        high_rate_movie_high_relevant_tag_count(movieId).sql
        high_rating_movies_genres.csv
        high_rating_movies_genres.sql
        movieId-count(rating)_mean(rating).sql
        movieId-count(rating)_mean_rating.csv
        rating-count(rating).csv
        rating-count(rating).sql
```

## Result:
