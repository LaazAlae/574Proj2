import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import pdb
# Problem 1
########################################################################################################################################################
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 

    #identify unique classes in y w goal of giving all distinct class labels
    classes = np.unique(y)
    # breakpoint()
    # print(X)
    # print(y)
    # print(classes)
    # d nbr of features k nbr of classes
    d = X.shape[1]
    k = len(classes)
    #Initialize a means covmat
    means = np.zeros((d, k))
    covmat = np.zeros((d, d))

    # total nbr training examples
    N = X.shape[0]
    
    #Loop over each class compute its mean
    for idx, c in enumerate(classes):
        #select all rows in X where  label =  current class c
        Xc = X[y.flatten() == c]
        #compute the mean of the selected rows for each feature
        means[:, idx] = np.mean(Xc, axis=0)
        #compute the difference between each sample and the computed mean
        diff = Xc - means[:, idx]
        # accumulate the covariance contribution from these samples
        covmat += diff.T @ diff
    #divide by the total number of samples to obtain the pooled covariance matrix
    covmat /= N
    
    return means, covmat



def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    #identify unique classes from y
    classes = np.unique(y)
    d = X.shape[1]
    k = len(classes)
    # initialize means covmat
    means = np.zeros((d, k))
    covmats = []
    
    #loop through each class to compute mean and classs covmat
    for idx, c in enumerate(classes):
        #extract c training examples
        Xc = X[y.flatten() == c]
        # compute store c mean
        means[:, idx] = np.mean(Xc, axis=0)
        #calculate diff each sample and class mean
        diff = Xc - means[:, idx]
        # c cov by multiplying transpose of differences w differences then div by nbr of samples
        covmat = (diff.T @ diff) / Xc.shape[0]
        #Append the cov matrix to list
        covmats.append(covmat)

    return means,covmats



def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    #check if ytest all zeros
    if np.unique(ytest).size == 1 and np.unique(ytest)[0] == 0:
        # If so assume class labels 0, 1,... nbr classes - 1
        classes = np.arange(means.shape[1])
    else:
        classes = np.unique(ytest)
    
    #invert shared cov matrix to compute distance
    invCov = np.linalg.inv(covmat)
    N = Xtest.shape[0]  #nbr test examples
    k = means.shape[1]  #number of classes
    #initialize perdictin vector
    ypred = np.zeros((N, 1))
    
    #loop through each test example
    for i in range(N):
        x = Xtest[i, :]  
        distances = np.zeros(k)  
        # Calculate distance from x to each class mean
        for j in range(k):
            diff = x - means[:, j]
            distances[j] = diff.T @ invCov @ diff  # squared distance
        # find class with the min distance.
        idx = np.argmin(distances)
        #assign the predicted class label
        ypred[i] = classes[idx]
    
    # compute accuracy as the percentage of test examples correctly classified
    acc = np.mean(ypred.flatten() == ytest.flatten()) * 100

    return acc,ypred



def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    #check if ytest contains all zero
    if np.unique(ytest).size == 1 and np.unique(ytest)[0] == 0:
        classes = np.arange(means.shape[1])
    else:
        classes = np.unique(ytest)
    
    N = Xtest.shape[0]  # nbr test examples
    k = means.shape[1]  # nbr classes
    ypred = np.zeros((N, 1))
    
    for i in range(N):
        x = Xtest[i, :]
        vals = np.zeros(k)  #array to store discriminant values for each class
        #compute the discriminant function for each class
        for j in range(k):
            cov = covmats[j]  #get each class cov mat 
            diff = x - means[:, j]
            invCov = np.linalg.inv(cov)       #inverse of cov mat 
            detCov = np.linalg.det(cov)         #Determinant of cov mat 
            #discriminant function 0.5*log(det(cov)) + 0.5*(x-mu)^T*inv(cov)*(x-mu)
            vals[j] = 0.5 * np.log(detCov) + 0.5 * (diff.T @ invCov @ diff)
        #predict class corresponding to the smallest discriminant value.
        idx = np.argmin(vals)
        ypred[i] = classes[idx]
    
    #calculate classification accuracy.
    acc = np.mean(ypred.flatten() == ytest.flatten()) * 100

    return acc,ypred
########################################################################################################################################################












# Problem 2
########################################################################################################################################################
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD   

    #normal equation to compute regression weights "w = (X^T * X)^(-1) * X^T * y"
    w = np.linalg.inv(X.T @ X) @ (X.T @ y)   

    return w


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD

    #get predicted values to calculate error "diff between predicted and actual values"
    error = ytest - (Xtest @ w)
    mse = np.mean(error**2)
    
    return mse
########################################################################################################################################################








def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1      
    # IMPLEMENT THIS METHOD
    # After taking the derivative we get w = (XᵀX + λI)^(-1)Xᵀy

    d = X.shape[1] # Number of features/columns
    I = np.eye(d) # Identity matrix of size d x d
    w = np.linalg.inv(X.T @ X + lambd*I) @ (X.T @ y)
                                                    
    return w


def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  
    # The gradient for this problem is: −Xᵀ(y−Xw)+λw
    # IMPLEMENT THIS METHOD

    w = w.reshape(-1, 1)  # Ensure w is column vector
    error = 0.5 * ((y - X @ w).T @ (y - X @ w) + lambd * (w.T @ w))

    # Convert scalar to float
    error = float(error)
    error_grad = -X.T @ (y - X @ w) + lambd * w
    # Flatten gradient for scipy.minimize compatibility
    error_grad = error_grad.flatten()

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    x = x.flatten()  
    N = x.shape[0]
    Xp = np.zeros((N, p+1))
    for i in range(p+1):
        Xp[:, i] = x**i
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones(X_i.shape[1])
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # Optimized value found in problem 3 and 4
print(f"Optimal lambda value: {lambda_opt}")
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
