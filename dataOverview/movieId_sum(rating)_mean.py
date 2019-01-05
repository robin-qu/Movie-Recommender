"""
Movie Recommender Project
Hongbin Qu
This program uses SQL query statement to find out
how many ratings that each movie got and their mean rating
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt

connection = sqlite3.connect("../dataset/small_dataset.db")
cursor = connection.cursor()
cursor.execute("select movieId, sum(rating)/count(rating) as mean_rating\
                             from ratings\
                             group by movieId\
                             order by mean_rating;")
movieId_countRate_meanRate= cursor.fetchall()
cursor.close()
connection.close()
movieId= [x[0] for x in movieId_countRate_meanRate]
mean_rate= [x[1] for x in movieId_countRate_meanRate]
y_pos = np.arange(len(mean_rate))

plt.figure(1)
plt.hist(mean_rate, bins=50, normed=True, alpha=0.5,)
plt.xlabel('Movie ID')
plt.ylabel('Mean Rating')
plt.title('The distribution of mean ratings')
plt.show()
