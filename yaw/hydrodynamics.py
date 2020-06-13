import numpy as np

def getInputData():
    dataFile = np.load("input_cw_ccw.npy")
    
    x,X,y,Y,z,Z = [],[],[],[],[],[]
    """
    GLOBAL FRAME
    x - position
    X - surge velocity
    y - y direction position
    Y - sway velocity
    z - z direction position
    Z - heave velocity
    """
    pi,pi1,theta,theta1,shi,shi1 = [],[],[],[],[],[]
    
    """
    GLOBAL FRAME
      - roll position - pi
     - roll angular velocity - pi1
    θ  - pitch position - theta
    θ1 - pitch angular velocity - theta1
    Ψ  - yaw position - shi
    Ψ1 - yaw angular velocity - shi1
    
    """
    epn1, epn2, epn3, epn4, epn5, epn6 = [],[],[],[],[],[]    
    """
    Body Frame
    1  - x co-ordinate
    2  - y co-ordinate
    3  - z co-ordinate
    4  - roll position
    5  - pitch position
    6  - yaw position
    """
    u,v,w,p,q,r = [],[],[],[],[],[]
    """
    u - surge velocity
    v - sway velocity
    w - heave velovity
    p - roll angular velocity
    q - pitch angular velocity
    r - yaw angular velocity
    
    """
    rudderAngle, rpm = [],[] #propeller speeed#
    """
    """
    
    for line in dataFile:
        TA = line #TA = temporary array#
        x.append(float(TA[0]))
        X.append(float(TA[1]))
        y.append(float(TA[2]))
        Y.append(float(TA[3]))
        z.append(float(TA[4]))
        Z.append(float(TA[5]))
        pi.append(float(TA[6]))
        pi1.append(float(TA[7]))
        theta.append(float(TA[8]))
        theta1.append(float(TA[9]))
        shi.append(float(TA[10]))
        shi1.append(float(TA[11]))
        epn1.append(float(TA[12]))
        u.append(float(TA[13]))
        epn2.append(float(TA[14]))
        v.append(float(TA[15]))
        epn3.append(float(TA[16]))
        w.append(float(TA[17]))
        epn4.append(float(TA[18]))
        p.append(float(TA[19]))
        epn5.append(float(TA[20]))
        q.append(float(TA[21]))
        epn6.append(float(TA[22]))
        r.append(float(TA[23]))
        rudderAngle.append(float(TA[24]))
        rpm.append(float(TA[25]))
    return x,X,y,Y,z,Z,pi,pi1,theta,theta1,shi,shi1,epn1, epn2, epn3,epn4, epn5, epn6,u,v,w,p,q,r,rudderAngle, rpm


def velocity(u,v):
    U = []
    for i in range(len(u)):
        temp = (u[i] + v[i])**0.5
        U.append(temp)
    return U[1:]
  
def rudderAngleChange(rudderAngle):
    D = []
    for i in range(1,len(rudderAngle)):
        temp = rudderAngle[i]-rudderAngle[i-1]
        D.append(temp)
    return D

def delta_val(u):
    D =[]
    for i in range(1,len(u)):
        temp = u[i]-u[i-1]
        D.append(temp)
    return D
"""
surgeXC = first governing equation
swayYC = second governing equation
yawRC = third governing equation

"""

def Create_Surgecomponents(u,v, yawRate, RAC, TV):
    Surge_Components = []
    for i in range(len(u)):
        c1 = u[i]
        c2 = u[i]*TV[i]
        c3 = u[i]**2
        c4 = (u[i]**3)/TV[i]
        c5 = v[i]**2
        c6 = yawRate[i]**2
        c7 = (RAC[i]**2)*(TV[i]**2)
        c8 = (RAC[i]**2)*u[i]*TV[i]
        c9 = v[i]*yawRate[i]
        c10 = v[i]*RAC[i]*TV[i]
        c11 = v[i]*RAC[i]*u[i]
        c12 = u[i]*(v[i]**2)/TV[i]
        c13 = u[i]*(yawRate[i]**2)/TV[i]
        c14 = u[i]*v[i]*yawRate[i]/ TV[i]
        c15 = yawRate[i]*RAC[i]*TV[i]
        c16 = u[i]*yawRate[i]*RAC[i]
        c17 = TV[i]**2
        temp = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17]
        Surge_Components.append(temp)
    Surge_Components.pop()
    return Surge_Components


def Create_Swaycomponents(U,V,R,RAC,TV):
    Sway_Components = []
    for i in range(len(U)):
        c1 = V[i]
        c2 = TV[i]**2
        c3 = U[i]*TV[i]
        c4 = U[i]**2
        c5 = V[i]*TV[i]
        c6 = R[i]*TV[i]
        c7 = RAC[i]*(TV[i]**2)
        c8 = (V[i]**3)/TV[i]
        c9 = (RAC[i]**3)*(TV[i]**2)
        c10 = (V[i]**2)*R[i]/TV[i]
        c11 = (V[i]**2)*RAC[i]
        c12 = V[i]*(RAC[i]**2)*TV[i]
        c13 = RAC[i]*U[i]*TV[i]
        c14 = V[i]*U[i]
        c15 = R[i]*U[i]
        c16 = RAC[i]*(U[i]**2)
        c17 = (R[i]**3)/TV[i]
        c18 = V[i]*(R[i]**2)/TV[i]
        c19 = V[i]*(U[i]**2)/TV[i]
        c20 = R[i]*(U[i]**2)/TV[i]
        c21 = R[i]*(RAC[i]**2)*TV[i]
        c22 = (R[i]**2)*RAC[i]
        c23 = R[i]*V[i]*RAC[i]
        temp=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,
              c18,c19,c20,c21,c22,c23]
        Sway_Components.append(temp)
    Sway_Components.pop()
    return Sway_Components

def Create_Yawcomponents(U,V,R,RAC,TV):
    Yaw_Components = []
    for i in range(len(U)):
        c1 = R[i]
        c2 = U[i]**2
        c3 = U[i]*TV[i]
        c4 = U[i]**2
        c5 = V[i]*TV[i]
        c6 = R[i]*TV[i]
        c7 = RAC[i]*(TV[i]**2)
        c8 = V[i]**3 / TV[i]
        c9 = (RAC[i]**3)*(TV[i]**2)
        c10 = (V[i]**2)*R[i]/TV[i]
        c11 = (V[i]**2)*RAC[i]
        c12 = V[i]*(RAC[i]**2)*TV[i]
        c13 = RAC[i]*U[i]*TV[i]
        c14 = V[i]*U[i]
        c15 = R[i]*U[i]
        c16 = RAC[i]*(U[i]**2)
        c17 = (R[i]**3)/TV[i]
        c18 = V[i]*(R[i]**2)/TV[i]
        c19 = V[i]*(U[i]**2)/U[i]
        c20 = R[i]*(U[i]**2)/TV[i]
        c21 = R[i]*(RAC[i]**2)*TV[i]
        c22 = (R[i]**2)*RAC[i]
        c23 = R[i]*V[i]*RAC[i]
        temp=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,
              c17,c18,c19,c20,c21,c22,c23]
        Yaw_Components.append(temp)
    Yaw_Components.pop()
    return Yaw_Components


def EqL1(U):# left hand side of the equation
    Unext = []
    for i in range(len(U) -1):
        temp = U[i+1]
        Unext.append(temp)
    return Unext

def EqL2(V):
    Vnext = []
    for i in range(len(V) -1):
        temp = V[i+1]
        Vnext.append(temp)
    return Vnext

def EqL3(R):
    Rnext = []
    for i in range(len(R) -1):
        temp = R[i+1]
        Rnext.append(temp)
    return Rnext

"""
creating hydrodynamic components

"""



x,X,y,Y,z,Z,pi,pi1,theta,theta1,shi,\
shi1,epn1, epn2, epn3,epn4, epn5, epn6,u,v,w,p,q,r,rudderAngle, rpm = getInputData() 

TV = velocity(u,v)
RAC = rudderAngleChange(rudderAngle)
del_u = delta_val(u)
del_v = delta_val(v)
del_r = delta_val(r)
surgeXCt = Create_Surgecomponents(del_u,del_v,del_r,RAC,TV)
yawRCt = Create_Yawcomponents(del_u,del_v,del_r,RAC,TV)
swayYCt = Create_Swaycomponents(del_u,del_v,del_r,RAC,TV)
nUt = EqL1(del_u)
nVt = EqL2(del_v)
nRt = EqL3(del_r)

# print(len(del_u),len(del_v),len(del_r))
# print(len(RAC))
# print(len(TV))
# print(len(yawRC))

# print(len(nV))


# a = surgeXCt[0:5490]
# b = surgeXCt[5707:]
# c = yawRCt [0:5490]
# d = yawRCt [5707:]
# e = swayYCt[0:5490]
# f = swayYCt[5707:]
# aa = nUt[0:5490]
# bb = nUt[5707:]
# cc = nVt[0:5490]
# dd = nVt[5707:]
# ee = nRt[0:5490]
# ff = nRt[5707:]

surgeXC = surgeXCt[0:5490]
yawRC = yawRCt[0:5490]
swayYC = swayYCt[0:5490]
nU = nUt[0:5490]
nV = nVt[0:5490]
nR = nRt[0:5490]


    
    