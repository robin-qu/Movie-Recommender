"""
Movie Recommender Project
Hongbin Qu
This program implements deep learning method to 
predict movie rating
"""

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torch.utils.data as datas

from sklearn.model_selection import train_test_split
import scipy.sparse
from torch.autograd import Variable

data= pd.read_csv('../dataset/ratings.csv')

USER_NUM = data.userId.unique().shape[0]
MOVIE_NUM = data.movieId.unique().shape[0]
userId=list(data["userId"].values)
movieId=list(data["movieId"].values)

def reshape_r_matrix(data,index,columns,values,replace_value):
     reshaped=data.pivot(index,columns,values).fillna(replace_value) 
     return reshaped
 
shaped_data=reshape_r_matrix(data,"userId","movieId","rating",0)

train, test = train_test_split(shaped_data, test_size=0.2, random_state=23)
train = scipy.sparse.coo_matrix(train)
test = scipy.sparse.coo_matrix(test)

#  process the data to usable form
class Interactions(datas.Dataset):
    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]
    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val
    def __len__(self):
        return self.mat.nnz


trainloader = torch.utils.data.DataLoader(Interactions(train), batch_size=50,shuffle=True, num_workers=10)
testloader = torch.utils.data.DataLoader(Interactions(test), batch_size=50,shuffle=False, num_workers=10)

class Net(nn.Module):
    def __init__(self, num_of_user, num_of_movie, n,M):
        super(Net, self).__init__()
        self.user_layer = nn.Embedding(num_of_user, n,sparse=True)
        self.movie_layer= nn.Embedding(num_of_movie, n,sparse=True)
        self.dpot1=nn.Dropout(0.3)
        self.dpot2=nn.Dropout(0.1)
        self.dpot3=nn.Dropout(0.02)
        self.fc1 = nn.Linear(n*2, M)
        self.fc2 = nn.Linear(M, int(M/2))
        self.fc3=nn.Linear(int(M/2),1)
        #self.dropout = nn.Dropout(p=0.02)
        


    def forward(self, userId,movieId):
        
         user_eb= self.user_layer(userId)
         movie_eb = self.movie_layer(movieId)
         #x=self.u_bia(userId)+self.m_bia(movieId)
         x = torch.cat([user_eb, movie_eb], 1)
         x=self.dpot1(x)
         x =self.dpot2(F.relu(self.fc1(x)))
         x = self.dpot3(F.relu(self.fc2(x)))
         x=self.fc3(x)

         return x

    
    def predict(self, userId, movieId):
        # return the score
        pred = self.forward(userId, movieId)
        return pred


#net = Net(USER_NUM,MOVIE_NUM,200,60)

def train_and_test(Ir,momentum,trainloader,testloader,epoch):
    criterion = nn.MSELoss() 
    optimizer = optim.SGD(net.parameters(),Ir,momentum)
    train_accu=[]
    test_accu=[]
    all_pred_array=[]
    all_exact_array=[]
    for epoch in range(epoch):  # loop over the dataset multiple times
        print("Epoch:",epoch)
        running_loss = 0.0
        total = 0
        pred_array=np.empty(0)
        exact_array=np.empty(0)
        for  i, ((row, col), val) in enumerate(trainloader, 0):
            row = Variable(row.long())
            val = Variable(val).float()
           # print(row.shape)
           # print(val.shape)
            col = Variable(col.long())
            optimizer.zero_grad()
            outputs = net(row,col)
            loss = criterion(outputs[:,0], val)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total=total+row.size()[0]
        value=running_loss/total
        print("train_accuracy:",value)
        train_accu.append(value)
        test_loss=0.0
        total_test = 0
        with torch.no_grad():
              for  i, ((row, col), val) in enumerate(testloader, 0):
                   row = Variable(row.long())
                   val = Variable(val).float()
                   col = Variable(col.long())
                   outputs = net(row,col)
                   pred_array=np.concatenate((pred_array,outputs[:,0].numpy()),axis=0)
                   exact_array=np.concatenate((exact_array,val.numpy()),axis=0)
                   loss=criterion(outputs[:,0], val)
                   test_loss += loss.item()
                   total_test=total_test+row.size()[0]
        value=test_loss/total_test
#        for v in range(len(exact_array)):
#          if pred_array[v]>3:
#              pred_array[v]=1
#          else:
#              pred_array[v]=0
#          if exact_array[v]>3:
#              exact_array[v]=1
#          else:
#              exact_array[v]=0
        all_pred_array.append(pred_array)
        all_exact_array.append(exact_array)
        print("pred size:",len(all_pred_array))
        print("exact size:",len(all_exact_array))
        print("test_accuracy:",value)
        test_accu.append(value)
    print('Finished Training')
    return train_accu,test_accu,all_pred_array,all_exact_array

#dtr,te=train_and_test(0.001,0.4,trainloader,testloader,3)


def generate_parameters(n):
    Ir=[]
    momentum=np.random.uniform(low=0.4, high=0.99, size=n)
    Ir_pow=np.random.randint(low=-7, high=-1, size=n)
    for v in range(n):
         Ir.append(10.0**(Ir_pow[v]))
    N=np.arange(10,80,10)
    ind=np.random.randint(low=0, high=len(N), size=n)
    N=N[ind]
    M=[]
    for v in N:
        m=np.random.uniform(v*2*0.3,v*2*0.8,1)
        m=int(m)
        M.append(m)
    
    
    return Ir,momentum,N,M
         

All_train_accuracy=[]
All_test_accuracy=[]
n=20
Ir,momentum,N,M=generate_parameters(n)
for v in range(n):
    print("Trial:",v)
    print("Ir=:",Ir[v])
    print("Momentum:",momentum[v])
    print("M:",M[v])
    print("N:",N[v])
    net=Net(USER_NUM,MOVIE_NUM,N[v],M[v])
    train_accu,test_accu=train_and_test(Ir[v],momentum[v],trainloader,testloader,30)
    
    All_train_accuracy.append(train_accu)
    All_test_accuracy.append(test_accu)
    
    
x=np.arange(1,30+1)
for v in range(n):
    plt.figure()
    plt.plot(x,All_test_accuracy[v],label='Test loss')
    plt.plot(x,All_train_accuracy[v],label='Training loss')
    plt.legend(loc='upper right')
    plt.title("Training and test MSE")
    plt.xlabel("Iteration Number")
    plt.ylabel("MSE_loss")
    plt.show()

mse = np.mean((all_exact_array[29]- all_pred_array[29])**2)


all_mse=[]
for v in range(30):
    mse = np.mean((all_exact_array[v]- all_pred_array[v])**2)
    all_mse.append(mse)