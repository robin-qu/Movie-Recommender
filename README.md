Collaborative Filtering Movie Recommender System
================================================


This project implements collaborative filtering recommendation system in memory-based approaches and model based approaches, specifically, user-based collaborative filtering, item-based collaborative filtering, clustering-based approach (k nearest neighbors), matrix factorization approach (nonnegative matrix factorization), alternating gradient descent, and neuron network approach, with data taken from MovieLens dataset, to predict user ratings for movies and recommend movies to users based on the prediction. Root mean square error (RMSE) and area under the curve (AUC) are used as the evaluation metrics.

Database website: https://grouplens.org/datasets/movielens/

### Project structure:

DataPreviews directory: Includes how the data is retrieved, merged and analyzed, which are implemented in python, as well as plots and tables that are generated about the dataset features.

SQL_statement directory: Contains SQL statements that extract, aggregate and merge the data and the output tables.

doc directory: Contains project report and poster.

related_tutorial directory: Contains usefull papers and tutorials.

small_dataset.db: We built this file in a local database software, in order to analyze by using sql statements

All .py files: All calculation, analyze processes and learning algorithms are implemented in these .py files.
