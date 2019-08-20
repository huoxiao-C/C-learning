import numpy as np
import math


#initial 
X = np.array([[1,0,0],[1,2,2],[1,2,0],[1,3,0]]).astype(np.float32)
y = np.array([[0],[0],[1],[1]]).astype(np.float32)
w = np.array([[0],[0],[0]]).astype(np.float32)
num_Neg = 0

#Negative gradient
while(True):
    sigma = 1/(1+np.exp(np.dot(-X,w)))
    g = np.dot(X.transpose(),sigma-y)
    if(np.linalg.norm(g)<1e-5):
        break;
    w = w-g
    num_Neg += 1
    print(num_Neg)
w_neg = w
print("w_neg",w_neg)
#Newton method 
w = np.array([[0],[0],[0]]).astype(np.float32)
num_New = 0
while(True):
    sigma = 1/(1+np.exp(np.dot(-X,w)))
    g = np.dot(X.transpose(),sigma-y)
    if(np.linalg.norm(g)<1e-5):
        break;
    D = np.diag((sigma*(1-sigma)).flatten())
    H = np.dot(X.transpose(),np.dot(D,X))
    w = w-np.dot(np.linalg.inv(H),g)
    num_New += 1
    print(num_New)
w_newton = w   
print("w_newton",w_newton)
#BFGS method
num_BFGS = 1
invH = np.eye(3)
I = np.eye(3)
w = np.array([[0],[0],[0]]).astype(np.float32)
sigma = 1/(1+np.exp(np.dot(X,w)))
g = np.dot(X.transpose(),sigma-y)
while(np.linalg.norm(g)>1e-5):
    print("norm",np.linalg.norm(g))
    dk = np.dot(invH,g)
    w_new = w-dk
    print("temp= ",(w_new,w))
    s = -dk
    sigma = 1/(1+np.exp(np.dot(-X,w_new)))

    g_new = np.dot(X.transpose(),sigma-y)
    yB = g_new-g
    syBT = np.dot(s,yB.transpose())
    yBTs = np.dot(yB.transpose(),s)
    yBsT = syBT.transpose()
    ssT = np.dot(s,s.transpose())
    invH = np.dot((I-syBT/yBTs),np.dot(invH,(I-yBsT/yBTs)))+ssT/yBTs
    g = g_new
    w = w_new
    num_BFGS += 1
    print(num_BFGS)
w_BFGS = w
print(w_neg)
print(w_newton)
print(w_BFGS)
