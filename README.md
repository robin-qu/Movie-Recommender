Collaborative Filtering Movie Recommender System
================================================


This project performed collaborative filtering in user-item approach, item-item approach, knn, and unsupervised knn on MovieLen database to recommend movies for users based on ratings. The original database is large, for convenience, a small version of the dataset was used in this project.

Database website: https://grouplens.org/datasets/movielens/

This repository include:

DataPreviews: A folder include how we managed, merged and analyze the data, they are implemented in .py, and there are also plots and tables that were generated about the dataset features.

SQL_statement: In order to have an entire view of the big 20M dataset, we use SQL statement to extract, aggregate and merge the data. This folder contains all those SQL statements and the output tables.

related_tutorial: Before working on this project, we found some usefull papers and tutorial ppts from website, we collect them there.

small_dataset.db: We built this file in a local database software, in order to analyze by using sql statements

All .py files: All calculation, analyze processes and learning algorithms are implemented in these .py files.
