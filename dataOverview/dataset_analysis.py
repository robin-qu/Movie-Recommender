"""
Movie Recommender Project
Hongbin Qu
This program examines the basic infomation of the datasets
and generates some plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

links = '../dataset/links.csv'
movies = '../dataset/movies.csv'
ratings = '../dataset/ratings.csv'
tags = '../dataset/tags.csv'

df_l = pd.read_csv(links)
df_m = pd.read_csv(movies)
df_r = pd.read_csv(ratings)
df_t = pd.read_csv(tags)

def build_R(users_array, movies_array, movies_dict, ratings_array):
    '''
    Given an array of userids, an array of movieids and 
    a dictionary mapping from movieids to indices and 
    corresponding rating value, builds and returns an 
    array represents the R matrix
    '''
    n = max(users_array)
    m = len(movies_dict.keys())
    R = np.zeros([n, m])
    for i in range(len(ratings_array)):
        row = users_array[i] - 1
        col = movies_dict[movies_array[i]]
        R[row, col] = ratings_array[i]
    return R

# Dataset overview
users_array = df_r['userId'].values
movies_array = df_r['movieId'].values
ratings_array = df_r['rating'].values
unique_user_num = df_r['userId'].nunique()
unique_movie_num = df_r['movieId'].nunique()
unique_movie_list = sorted(list(set(movies_array)))
movie_idx_dict = {}
for i in range(unique_movie_num):
    movie_idx_dict[unique_movie_list[i]] = i
max_usersid = max(users_array)
max_moviesid = max(movies_array)

# Builds R matrix
R = build_R(users_array, movies_array, movie_idx_dict, ratings_array)

# Calculates sparsity of R matrix
W = (R != 0)
W[W == True] == 1
W[W == False] == 0
sparsity = np.sum(W) / (len(W[0]) * len(W[:, 0]))

def plot_frequency_of_ratings():
    '''
    Plots frequency of rating values
    '''
    rating_val = np.arange(0.5, 5.1, 0.5)
    frequency = np.zeros(10)
    for i in range(len(rating_val)):
        frequency[i] = len(np.where(ratings_array == rating_val[i])[0])
    plt.figure(1)
    plt.bar(rating_val, frequency, width = 0.4)
    plt.xticks(rating_val)
    plt.xlabel("Rating value")
    plt.ylabel("Number of ratings")
    plt.title("Frequency of Rating Values")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution_of_movie():
    '''
    Distribution of ratings among movies
    '''
    rating_num_array_m = np.zeros(max_moviesid)
    for i in range(max_moviesid):
        if i in movie_idx_dict.keys():
            rating_num_array_m[i] = np.sum(W[:, movie_idx_dict[i]])
        else:
            rating_num_array_m[i] = 0
    plt.figure(2)
    plt.scatter(range(max_moviesid), rating_num_array_m, s = 1)
    plt.xlabel("MovieID")
    plt.ylabel("Number of ratings")
    plt.title("Distribution of Ratings among Movies")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution_of_user():
    '''
    Distribution of ratings among users
    '''
    rating_num_array_u = np.zeros(unique_user_num)
    for i in range(unique_user_num):
        rating_num_array_u[i] = np.sum(W[i])
    plt.figure(3)
    plt.scatter(range(unique_user_num), rating_num_array_u, s = 1)
    plt.xlabel("UserID")
    plt.ylabel("Number of ratings")
    plt.title("Distribution of Ratings among Users")
    plt.tight_layout()
    plt.show()

def plot_rating_variance_of_movie():
    '''
    Distribution of movie numbers among rating variance
    '''
    rating_variance = np.zeros(unique_movie_num)
    for i in range(unique_movie_num):
        rating_variance[i] = np.var(R[:, i])
    plt.figure(4)
    plt.hist(rating_variance)
    plt.xlabel("Rating Variance")
    plt.ylabel("Number of Movies")
    plt.title("Distribution of movie numbers among rating variance")
    plt.tight_layout()
    plt.show()

def print_info():
    '''
    Prints the information of the dataset
    '''    
    print('File "links.csv" information:', df_l.info())
    print()
    print('File "links.csv" information:', df_m.info())
    print()
    print('File "links.csv" information:', df_r.info())
    print()
    print('File "links.csv" information:', df_t.info())
    print()
    print('The number of unique users is', unique_user_num)
    print('The maximum userId is', max_usersid)
    print('The number of unique movies is', unique_movie_num)
    print('The maximum movieId is', max_moviesid)
    check_duplicated = df_r.duplicated(subset = ['userId','movieId']).sum()
    if check_duplicated == 0:
        print('No duplicated rating.')
    else:
        raise ValueError('Duplicated ratings exist in dataset!')
    print()
    print('R matrix is a', R.shape[0], 'by', R.shape[1], 'matrix.')
    print('The sparsity of matrix R is', sparsity)  

def main():
    print_info()
    plot_frequency_of_ratings()
    plot_rating_distribution_of_movie()
    plot_rating_distribution_of_user()
    plot_rating_variance_of_movie()
    
if __name__ == "__main__":
    main()