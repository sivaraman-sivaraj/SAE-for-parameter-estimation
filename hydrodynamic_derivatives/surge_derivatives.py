# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:59:46 2020

@author: srama
"""


import numpy as np
import time
from tabulate import tabulate

X = np.load('surge_regression_coeff.npy')


print(X)


"""
start of declaration of ship features

"""
h = 0.05 # time step size
L = 7.0 # length of ship
Xg = 0.25 # Longitutional co-ordinate of ship center of gravity 
m = 3.27*1025 # mass of ship
IzG = m*((0.25*L)**2) # Moment of inertia of ship around center of gravity
lenthofShip = 7.0


Xau= (2*174.994)/(1025*(7**3)) # accelaration derivative of surge force with respect to u 

Yav = (2*1702.661)/(1025*(7**3)) # accelaration derivative of sway force with respect to v
Yar = (2*1273.451)/(1025*(7**4)) # accelaration derivative of sway force with respect to r

Nav = (2*1273.451)/(1025*(7**4)) # Yaw moment derivative with respect to sway velocity
Nar = (2*9117.302)/ (1025*(7**4)) # Yaw moment derivative with respect to rudder angle

S = ((IzG-Nar)*(m-Yav))-(((m*Xg)-Yar)*((m*Xg)-Nav))

"""
End of features description
"""
"""

surgre solution and derivatives
"""

def term1(h,L,m,Xau):
    return h/(L*(m-Xau))

CC1 = term1(h,L,m,Xau)
t1 = X[0][1:]

surge_derivatives = [i for i in map(lambda x : CC1*x, t1)]

print(" \n \n Derivatives components of surge equation: \n \n",surge_derivatives)
