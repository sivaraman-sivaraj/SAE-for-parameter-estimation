# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:06:45 2020

@author: Sivaraman Sivaraj, Suresh Rajendran
"""


print(__doc__)

import hydrodynamics as hdy

"""
TV, RAC, surgeXC, yawRC, swayYC, nU, nV,nR # importing variables from hydrodynamics

x,X,y,Y,z,Z,pi,pi1,theta,theta1,shi,\
shi1,epn1, epn2, epn3,epn4, epn5, epn6,u,v,w,p,q,r,rudderAngle, rpm
"""

import numpy as np, scipy as sp, matplotlib.pyplot as plt,datetime,time,random
import torch, torch.nn.functional as F
import torch.nn as nn
import datetime
import torch.optim as optim

start = time.time()


x = hdy.surgeXC
y = hdy.nU
print()
x_pt = x[1:10000]
y_pt = y[1:10000]


class SingleNet(nn.Module): 

    def __init__(self, D_in,D_out):
        super(SingleNet, self).__init__()
        
        self.fc1 = nn.Linear(D_in, D_out)
        
        
    def forward(self, x):
        
        x = torch.tanh(self.fc1(x))
        
        return x

net = SingleNet(17,1)
params =list( net.parameters())

np.random.seed(0)
a = np.random.randn(1, 17)

aa = torch.tensor(a)

aa = params[0]

def PreTraining(x,y,D_in,D_out,lr,n_epoch): #as we know D_in and D_out is same at network
# x - input, D_in - input dimension,lr - learning rate, n_epoch - number of epochs
    N = len(x)#batch size
    np.random.seed(0)
    net = SingleNet(D_in, D_out)#auto associative neural network part
    params =list( net.parameters())
    a = np.random.randn(1, 17)
    aa = torch.tensor(a)
    params[0] = aa
    criterion = nn.MSELoss()#mean square error
    optimizer = optim.Adam(net.parameters(), lr,betas = (0.01,0.03)) #gradiant update
    
    # n_epoch = 10 #n_epoch - number of epochs
    for epoch in range(n_epoch):
        
        lossM = []
        for i in range(N):
            input = torch.tensor(x[i])
            target = torch.tensor(round(y[i],7))
            # input = double(temp)
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()  # Does the update
            
            if loss>0.05:
                print(i)
            # if loss < 0.05:
            lossM.append(loss)
        # print(epoch)# to see which iteration is running
    print("Pre-training of", str(D_in),'*',str(D_out),'weight matrix has done')
    params = list(net.parameters())
    
    return params[0],lossM


weight, loss = PreTraining(x_pt, y_pt, 17, 1, 0.01, 2)

print(weight)


def errorplot(loss):
    plt.figure()
    plt.plot(loss,'g')
    plt.xlabel("number of iteration - batch size")
    plt.ylabel("Mean squared error")
    plt.title("Surge Equation plot")
    plt.savefig("surge_MSE.jpg")
    plt.show()


errorplot(loss)
with torch.no_grad():
    np.save('surge_regression_coeff.npy',weight.detach().numpy())  

end = time.time()
print("total time has taken to run this code is", (end - start),"seconds")
    






