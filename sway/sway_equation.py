"""
Created on Tue May 12 23:22:46 2020

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

def double(x):
    a = torch.zeros(1, len(x), dtype=torch.double)
    result = torch.empty(1, len(x))
    torch.add(a,x, out=result)
    return result

def float64(w,a,b):
    aa = torch.zeros(a,b, dtype = torch.float64)
    result = torch.empty(a,b)
    w.detach()
    w = torch.tensor(w)
    torch.add(aa,w, out=result)
    return result


class AANN(nn.Module): 

    def __init__(self, D_in,H,D_out):
        super(AANN, self).__init__()
        
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        
        
    def forward(self, x):
        
        x = (self.fc1(x))
        x = (self.fc2(x))
        
        return x


def PreTraining(x,D_in,H,D_out,lr,n_epoch): #as we know D_in and D_out is same at AANN
# x - input, D_in - input dimension, L1 - layer1 dimension, H - Bottleneck layer dimension
#L3 - layer3 dimension, lr - learning rate, n_epoch - number of epochs
    N = len(x)#batch size
    D_out = D_in
    net = AANN(D_in,H, D_out)#auto associative neural network part
    criterion = nn.MSELoss()#mean square error
    optimizer = optim.Adam(net.parameters(), lr,betas = (0.01,0.03)) #gradiant update
    
    # n_epoch = 10 #n_epoch - number of epochs
    for epoch in range(n_epoch):
        
        lossM = []
        for i in range(N):
            input = torch.tensor(x[i])
            target = input
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
    print("Pre-training of", str(D_in),'*',str(H),'weight matrix has done')
    params = list(net.parameters())
    
    return params[0],lossM

def Reduce_dimension(w1,x, D_in, H):
    reduced_feature = []
    a = w1.requires_grad_(False)
    
    for i in range(len(x)):
        b = torch.tensor(x[i])
        b = b.view(D_in,1)
        temp = a.mm(b)
        temp = temp.T
        reduced_feature.append(temp[0])
    return reduced_feature


def auto_encoders_reduction(x):
    """
    
    

    Parameters
    ----------
    x : input elements in yaw governing equation, size of 23.
    there will be 22 reduction trainings are there.

    Returns
    -------
    multiplication of all linear weight matrix final size of (23*1)
    
    w - weight matrix
    rf - reduced feature
    e - error

    """
    w23,e23 = PreTraining(x, 23, 22, 23, 0.01, 1)
    rf22 = Reduce_dimension(w23,x, 23,22)
    w22,e22 = PreTraining(rf22, 22, 21, 22, 0.01, 1)
    rf21 = Reduce_dimension(w22,rf22, 22,21)
    w21,e21 = PreTraining(rf21, 21, 20, 21, 0.01, 1)
    rf20 = Reduce_dimension(w21,rf21, 21,20)
    w20,e20 = PreTraining(rf20, 20,19,20, 0.01, 1)
    rf19 = Reduce_dimension(w20,rf20, 20,19)
    w19,e19 = PreTraining(rf19, 19,18,19, 0.01, 1)
    rf18 = Reduce_dimension(w19,rf19, 19,18)
    w18,e18 = PreTraining(rf18, 18,17,18, 0.01, 1)
    rf17 = Reduce_dimension(w18,rf18, 18,17)
    w17,e17 = PreTraining(rf17, 17,16,17, 0.01, 1)
    rf16 = Reduce_dimension(w17,rf17, 17,16)
    w16,e16 = PreTraining(rf16, 16,15,16, 0.01, 1)
    rf15 = Reduce_dimension(w16,rf16, 16,15)
    #pass break for see the code, reduced to 15 elements
    w15,e15 = PreTraining(rf15, 15,14,15, 0.01, 1)
    rf14 = Reduce_dimension(w15,rf15, 15,14)
    w14,e14 = PreTraining(rf14,14,13,14, 0.01, 1)
    rf13 = Reduce_dimension(w14,rf14, 14,13)
    w13,e13 = PreTraining(rf13, 13,12,13, 0.01, 1)
    rf12 = Reduce_dimension(w13,rf13, 13,12)
    w12,e12 = PreTraining(rf12, 12,11,12, 0.01, 1)
    rf11 = Reduce_dimension(w12,rf12, 12,11)
    w11,e11 = PreTraining(rf11, 11,10,11, 0.01, 1)
    rf10 = Reduce_dimension(w11,rf11, 11,10)
    #pass break for see the code, reduced to 10 elements
    w10,e10 = PreTraining(rf10, 10,9,10, 0.01, 1)
    rf9 = Reduce_dimension(w10,rf10, 10,9)
    w9,e9 = PreTraining(rf9,9,8,9, 0.01, 1)
    rf8 = Reduce_dimension(w9,rf9, 9,8)
    w8,e8 = PreTraining(rf8,8,7,8, 0.01, 1)
    rf7 = Reduce_dimension(w8,rf8, 8,7)
    w7,e7 = PreTraining(rf7,7,6,7, 0.01, 1)
    rf6 = Reduce_dimension(w7,rf7, 7,6)
    w6,e6 = PreTraining(rf6, 6,5,6, 0.01, 1)
    rf5 = Reduce_dimension(w6,rf6, 6,5)
    #pass break for see the code, reduced to 10 elements
    w5,e5 = PreTraining(rf5, 5,4,5, 0.01, 1)
    rf4 = Reduce_dimension(w5,rf5, 5,4)
    w4,e4 = PreTraining(rf4,4,3,4, 0.01, 1)
    rf3 = Reduce_dimension(w4,rf4, 4,3)
    w3,e3 = PreTraining(rf3,3,2,3, 0.01, 1)
    rf2 = Reduce_dimension(w3,rf3, 3,2)
    w2,e2 = PreTraining(rf2,2,1,2, 0.01, 1)
    rf1 = Reduce_dimension(w2,rf2,2,1)
    
  
    return w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,\
            w17,w18,w19,w20,w21,w22,w23, e2,e3,e4,e5,e6,e7,e8,\
                e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,\
                    e21,e22,e23,rf1



def non_linear_coefficients():#to get hydrodynamic derivatives
    w = torch.mm(w22, w23)
    w = torch.mm(w21,w)
    w = torch.mm(w20,w)
    w = torch.mm(w19,w)
    w = torch.mm(w18,w)
    w = torch.mm(w17,w)
    w = torch.mm(w16,w)
    w = torch.mm(w15,w)
    w = torch.mm(w14,w)
    w = torch.mm(w13,w)
    w = torch.mm(w12,w)
    w = torch.mm(w11,w)
    w = torch.mm(w10,w)
    w = torch.mm(w9,w)
    w = torch.mm(w8,w)
    w = torch.mm(w7,w)
    w = torch.mm(w6,w)
    w = torch.mm(w5,w)
    w = torch.mm(w4,w)
    w = torch.mm(w3,w)
    w = torch.mm(w2,w)
    return w
    
"""
computational part
"""

x = hdy.swayYC
y = hdy.nV



w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,\
            w17,w18,w19,w20,w21,w22,w23, e2,e3,e4,e5,e6,e7,e8,\
                e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,\
                    e21,e22,e23,rf1 = auto_encoders_reduction(x)
coeff = non_linear_coefficients()

np.savetxt('sway_regression_coeff.txt', coeff)
np.save('sway_regression_coeff.npy', coeff)


"""
plotting part
"""

def Errotplot1():
    plt.figure(figsize=(15,12))
    plt.subplot(331)
    plt.plot(e10)
    plt.title('error- sae(10*9)')
    plt.subplot(332)
    plt.plot(e9)
    plt.title('error- sae(9*8)')
    plt.subplot(333)
    plt.plot(e8)
    plt.title('error- sae(8*7)')
    plt.subplot(334)
    plt.plot(e7)
    plt.title('error- sae(7*6)')
    plt.subplot(335)
    plt.plot(e6)
    plt.title('error- sae(6*5)')
    plt.subplot(336)
    plt.plot(e5)
    plt.title('error- sae(5*4)')
    plt.subplot(337)
    plt.plot(e4)
    plt.title('error- sae(4*3)')
    plt.subplot(338)
    plt.plot(e3)
    plt.title('error- sae(3*2)')
    plt.subplot(339)
    plt.plot(e2)
    plt.title('error- sae(2*1)')
    plt.suptitle('Sway Mean Square Error Plot of Auto-encoders Training')
    plt.savefig('ErrorPlot1.jpg')
    plt.show()
    

def Errotplot2():
    plt.figure(figsize=(15,12))
    plt.subplot(331)
    plt.plot(e19)
    plt.title('error- sae(19*18)')
    plt.subplot(332)
    plt.plot(e18)
    plt.title('error- sae(18*17)')
    plt.subplot(333)
    plt.plot(e17)
    plt.title('error- sae(17*16)')
    plt.subplot(334)
    plt.plot(e16)
    plt.title('error- sae(16*15)')
    plt.subplot(335)
    plt.plot(e15)
    plt.title('error- sae(15*14)')
    plt.subplot(336)
    plt.plot(e14)
    plt.title('error- sae(14*13)')
    plt.subplot(337)
    plt.plot(e13)
    plt.title('error- sae(13*12)')
    plt.subplot(338)
    plt.plot(e12)
    plt.title('error- sae(12*11)')
    plt.subplot(339)
    plt.plot(e11)
    plt.title('error- sae(11*10)')
    plt.suptitle('Sway Mean Square Error Plot of Auto-encoders Training')
    plt.savefig('ErrorPlot2.jpg')
    plt.show()
    
def Errotplot3():
    plt.figure(figsize=(15,12))
    plt.subplot(221)
    plt.plot(e23)
    plt.title('error- sae(23*22)')
    plt.subplot(222)
    plt.plot(e22)
    plt.title('error- sae(22*21)')
    plt.subplot(223)
    plt.plot(e21)
    plt.title('error- sae(21*20)')
    plt.subplot(224)
    plt.plot(e20)
    plt.title('error- sae(20*19)')
    
    plt.suptitle('Sway Mean Square Error Plot of Auto-encoders Training')
    plt.savefig('ErrorPlot3.jpg')
    plt.show()

def Errorplot4(rf1,y):
    plt.figure()
    plt.scatter(rf1, y, edgecolors = 'green')
    plt.title("Predicted and Actual scatter Plot-Sway")
    plt.savefig("final_error.jpg")
    plt.show()
    
def rf1p():
    data = []
    for i in range(len(rf1)):
        data.append(float(rf1[0]))
    return data

rf1x = rf1p()

Errotplot1()
Errotplot2()
Errotplot3()
Errorplot4(rf1x,y)
    
end = time.time()
print("total time has taken to run this code is", (end - start),"seconds")
    
    
    
    






