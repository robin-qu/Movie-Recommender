"""
Movie Recommender Project
Hongbin Qu
This program implements a user based collaborative filtering 
algorithm using kNN algorithm with the help of surprise package
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.metrics import roc_curve, auc
from surprise import Reader, Dataset, accuracy
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection import cross_validate, train_test_split


def knn_cv(data):
    '''
    Calculate root mean square error using k nearest neighbor method
    with k starting from 2 to 50 in step sizes of 2
    10-folds cross-validation
    '''
    rmse = []
    k_list = range(2, 51, 2)
    print('Performing knn...')
    for k in k_list:
        print('k =', k)
        sim_options = {'name': 'cosine'}
        algo = KNNWithMeans(k=k, sim_options=sim_options, verbose=False)
        cv_result = cross_validate(algo, data, measures=['RMSE'], cv=10,
                                   verbose=False)
        rmse.append(np.mean(cv_result['test_rmse']))
    print('Completed!')
    return rmse, k_list

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

# Set starting timer
start = timer()

# Load dataset
reader = Reader(line_format='user item rating timestamp',
                sep=',', skip_lines=1)
data = Dataset.load_from_file('../dataset/ratings.csv', reader=reader)

# 10-fold cross validation
rmse, k_list = knn_cv(data)

# get optimal k
min_idx = rmse.index(min(rmse))
k_hat = k_list[min_idx]

# Training
trainset, testset = train_test_split(data, test_size=0.1)
sim_options = {'name': 'cosine'}
algo = KNNWithMeans(k=k_hat, sim_options=sim_options, verbose=False)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Plot Testing rmse
plt.figure(1)
plt.plot(k_list, rmse)
plt.xlabel('k')
plt.ylabel('Testing Root Mean Square Error')
plt.title('kNN: The Result of Average RMSE versus k')
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


# Set ending timer
end = timer()
print('time:', end - start)