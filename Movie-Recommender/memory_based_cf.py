"""
Movie Recommender Project
Hongbin Qu
This program implements a user based and a item based 
collaborative filtering algorithm
"""

import numpy as np
from math import sqrt
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from dataset_analysis import df_r, unique_user_num, unique_movie_num


def generate_rmatrix(df):
    '''
    Given a rating dataframe, generates a rating matrix whose row
    is userID and column is movieID and corresponding entry is the
    user's rating on the movie
    '''
    R = df.pivot(index = 'userId', columns = 'movieId',
                 values = 'rating').fillna(0).values
    appendix_row = np.zeros((unique_user_num - R.shape[0], R.shape[1]))
    R = np.concatenate((R, appendix_row), axis = 0)
    appendix_col = np.zeros((R.shape[0], unique_movie_num - R.shape[1]))
    R = np.concatenate((R, appendix_col), axis = 1)
    return R

def compute_rmse(prediction, R):
    '''
    Given predicted rating data and original rating data, computes
    and returns the root mean square error of the entries where the
    rating is specified
    '''
    prediction = prediction[R.nonzero()]
    R_specified = R[R.nonzero()]
    return sqrt(mean_squared_error(prediction, R_specified))

def userbased_predict(mean, R, S):
    '''
    Given the user average rating vector, rating matrix R and 
    similarity matrix S, fill in the entries with unknown rating,
    returns the prediction matrix under user-based model
    '''
    difference = np.copy(R)
    for i in range(difference.shape[0]):
        difference[i][np.where(difference[i] != 0)] -= mean[i]
    numerator = np.dot(S, difference)
    denominator = np.abs(S).sum(axis = 1)[:, np.newaxis]
    prediction = mean[:, np.newaxis] + numerator / denominator
    prediction[prediction < 0] = 0
    prediction[prediction > 5] = 5
    return prediction

def moviebased_predict(R, S):
    '''
    Given rating matrix R and similarity matrix S, fill in the
    entries with unknown rating, returns the prediction matrix
    under movie-based model
    '''
    prediction = np.dot(R, S) / np.abs(S).sum(axis = 1)
    prediction[prediction < 0] = 0
    prediction[prediction > 5] = 5
    return prediction

def svd_predict(R, d):
    '''
    Given rating matrix R and number of sigular values, performs
    sigular value decomposition and returns the prediction matrix
    '''
    U, S, V = svds(R, d)
    S_diag = np.diag(S)
    prediction = np.dot(U, np.dot(S_diag, V))
    prediction[prediction > 5] = 5
    prediction[prediction < 0] = 0
    return prediction
    
# Splits the dataset as training data and testing data
train_df, test_df = train_test_split(df_r, test_size = 0.1)

# Computes the R matrix on training and testing data
R_train = generate_rmatrix(train_df)
R_test = generate_rmatrix(test_df)

########### user-based collaborative filtering ################################

# Computes user-based similarity matrix
S_user = pairwise_distances(R_train, metric = 'cosine')

# Training set prediction
mean_user_rating = np.zeros(R_train.shape[0])
for i in range(R_train.shape[0]):
    mean_user_rating[i] = R_train[i][np.where(R_train[i] != 0)].mean()
prediction_train_ub = userbased_predict(mean_user_rating, R_train, S_user)

# Computes root mean squate error of training set
error_train_ub = compute_rmse(prediction_train_ub, R_train)
print('User_based training set prediction rmse:', error_train_ub)

# Testing Set Prediction
prediction_test_ub = userbased_predict(mean_user_rating, R_test, S_user)

# Computes root mean squate error of training set
error_test_ub = compute_rmse(prediction_test_ub, R_test)
print('User-based testing set prediction rmse:', error_test_ub)


########### item-based collaborative filtering ###############################

# Computes movie-based similarity matrix
S_movie = pairwise_distances(R_train.T, metric = 'cosine')

# Training set prediction
prediction_train_mb = moviebased_predict(R_train, S_movie)

# Computes root mean squate error of training set
error_train_mb = compute_rmse(prediction_train_mb, R_train)
print('Movie-based training set prediction rmse:', error_train_mb)

# Testing Set Prediction
prediction_test_mb = moviebased_predict(R_test, S_movie)

# Computes root mean squate error of training set
error_test_mb = compute_rmse(prediction_test_mb, R_test)
print('Movie-based testing set prediction rmse:', error_test_mb)

################ Singular Value Decomposition #################################
prediction_svd = svd_predict(R_train, 50)
error_train_svd = compute_rmse(prediction_svd, R_train)
error_test_svd = compute_rmse(prediction_svd, R_test)
print('0-filled SVD based training set prediction rmse:', error_train_svd)
print('0-filled SVD based training set prediction rmse:', error_test_svd)

train_mean = R_train[np.where(R_train != 0)].mean()
R_mean_train = np.copy(R_train)
R_mean_train[np.where(R_mean_train == 0)] = train_mean

prediction_mean_svd = svd_predict(R_mean_train, 50)
error_train_mean_svd = compute_rmse(prediction_mean_svd, R_train)
error_test_mean_svd = compute_rmse(prediction_mean_svd, R_test)
print('Mean-filled SVD based training set prediction rmse:',
      error_train_mean_svd)
print('Mean-filled SVD based training set prediction rmse:',
      error_test_mean_svd)
