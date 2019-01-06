"""
Movie Recommender Project
Hongbin Qu
This program implements a model based collaborative filtering
algorithm using non-negative matrix factorization algorithm with
the help of surprise package
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import NMF, accuracy
from surprise import Reader, Dataset
from timeit import default_timer as timer
from sklearn.metrics import roc_curve, auc
from surprise.model_selection import cross_validate, train_test_split


def nmf_cv(data):
    '''
    Calculate root mean square error using non negative matrix  
    factorization method with hidden variable r starting from
    2 to 50 in step sizes of 2 10-folds cross-validation
    '''
    rmse = []
    r_list = range(2, 51, 2)
    print('Performing nmf...')
    for r in r_list:
        print('r =', r)
        algo = NMF(n_factors=r, biased=False)
        cv_result = cross_validate(algo, data, measures=['RMSE'], cv=10, 
                                   verbose=False)
        rmse.append(np.mean(cv_result['test_rmse']))
    print('Completed!')
    return rmse, r_list

def threshold_target(threshold, test_target):
    '''
    Given a threshold value, loop over the test_target list, if a rating 
    is higher than the threshold, sets it as 1, otherwise sets it as 0, 
    stores in a new list and returns it.
    '''
    result = []
    for rating in test_target:
        if rating >= threshold:
            result.append(1)
        else:
            result.append(0)
    return result

def plot_roc(fpr, tpr):   
    '''
    Given lists of false positive rate and true positive rate, 
    plots the ROC curve
    '''
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.show()

def  print_genres(col, V_dict):
    '''
    Takes the V dictionary as input, prints the genres of movies for
    the col_th component
    '''
    Sorted_V_dict = sorted(V_dict.items(), key=lambda item:item[1][col],
                           reverse=True)
    print()
    print("Genres of the top 10 movies for %d th component are:" %col)
    for i in range(10):
        genres = movies_dict['genres']
        print(genres[movies_dict['movieID'].index(Sorted_V_dict[i][0])])


# Set starting timer
start = timer()

# Load dataset
reader = Reader(line_format = 'user item rating timestamp',
                sep = ',', skip_lines = 1)
data = Dataset.load_from_file('../dataset/ratings.csv', reader = reader)

# 10-fold cross validation
rmse, r_list = nmf_cv(data)

# get optimal r
min_idx = rmse.index(min(rmse))
r_hat = r_list[min_idx]

# Training and testing
trainset, testset = train_test_split(data, test_size=0.2)
algo = NMF(n_factors=r_hat, biased=False)
algo.fit(trainset)
U = algo.pu
V = algo.qi
predictions = algo.test(testset)
accuracy.rmse(predictions)

plt.figure(1)
plt.plot(r_list, rmse)
plt.xlabel('r')
plt.ylabel('Testing Root Mean Square Error')
plt.title('NMF: The Result of Average RMSE versus k')
plt.show()


# Plot ROC curve
test_target = []
test_score = []
for (_, _, true_rating, predict_rating, _) in predictions:
    test_target.append(true_rating)
    test_score.append(predict_rating)
    
binary_test = threshold_target(3, test_target)
fpr = dict()
tpr = dict()
fpr, tpr, thresholds = roc_curve(binary_test, test_score)
plot_roc(fpr, tpr)

### Interpretation ###
ratings = pd.read_csv('../dataset/ratings.csv')
ratings_dict = {'userID': list(ratings.userId),
                'movieID': list(ratings.movieId),
                'rating': list(ratings.rating)}
movies = pd.read_csv('../dataset/movies.csv')
movies_dict = {'movieID': list(movies.movieId),
                'title': list(movies.title),
                'genres': list(movies.genres)}

#build a movieID dict corrseponding to the index in ratings.csv
#key is the index, value is the movieID
movieid_dict={}
index=0
for i in range(len(ratings_dict['movieID'])):
    if(ratings_dict['movieID'][i] not in movieid_dict.values()):
        movieid_dict[index] = ratings_dict['movieID'][i]
        index = index + 1

#v matrix dict,, key is the movieID, values are the value in V matrix
V_dict={}
for i in range(len(V[:,0])):
    V_dict[movieid_dict[i]] = V[i,:]

print_genres(0, V_dict)

# Set ending timer
end = timer()
print('time:', end - start)