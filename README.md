Collaborative Filtering Movie Recommender System
================================================


This project implements collaborative filtering recommendation system in memory-based approaches and model based approaches, specifically, user-based collaborative filtering, item-based collaborative filtering, clustering-based approach (k nearest neighbors), matrix factorization approach (nonnegative matrix factorization), alternating gradient descent, and neuron network approach, with data taken from MovieLens dataset, to predict user ratings for movies and recommend movies to users based on the prediction. Root mean square error (RMSE) and area under the curve (AUC) are used as the evaluation metrics.

Database website: https://grouplens.org/datasets/movielens/

This repository include:

## DataPreviews: 
Includes how we retrieved, merged and analyze the data, they are implemented in .py, and there are also plots and tables that were generated about the dataset features.

## SQL_statement: 
In order to have an entire view of the big 20M dataset, we use SQL statement to extract, aggregate and merge the data. This folder contains all those SQL statements and the output tables.

## related_tutorial: 
Before working on this project, we found some usefull papers and tutorial ppts from website, we collect them there.

small_dataset.db: We built this file in a local database software, in order to analyze by using sql statements

All .py files: All calculation, analyze processes and learning algorithms are implemented in these .py files.
