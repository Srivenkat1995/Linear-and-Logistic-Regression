import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from math import sqrt
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.svm import SVC


print("Data sets are appended by running processing_Dataset.py")

data = pd.read_csv('Human_Feature_Concatenation.csv',index_col= False)
data = shuffle(data)
###############################################################################
##### Train,validate and Test Split###########################################

msk = np.random.rand(len(data)) < 0.8

train_set = data[msk]

remaining = data[~msk]

val = np.random.rand(len(remaining)) < 0.5

validate_set = remaining[val]

test_set = remaining[~val]

################ Linear Regression ############################################

X = np.array(train_set.iloc[:,2:20])   ##### Feature Vector

X0 = np.ones([X.shape[0],1])      ##### Base Feature

X = np.concatenate((X0,X),axis = 1)  ######## Concatenating base feature and Feature Vector 

Y = train_set.iloc[:,20:21].values        ####### Output Vector

theta = np.zeros([1,X.shape[1]])     ########## Corresponding theta Values
#################### Intializing the constants#####################

learning_rate = 0.03

no_of_iterations = 1000

####################################################################
# Computing cost Function ################

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

####################################################################
# Gradient Descent Function ###################

def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)

    return theta,cost

g,cost = gradientDescent(X,Y,theta,no_of_iterations,learning_rate) 



y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(g,X[i])


rms = sqrt(mean_squared_error(Y, y_prediction))
print("Root Mean Square Value of Human Feature Concatenation (Training)")
print(rms)


#######################################################################################################
############Logistic Regression ##############



def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros([features.shape[1],1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights = weights + np.dot(learning_rate,gradient)
        
           
    return weights

weights = logistic_regression(X, Y,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)


y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(weights[1:].T,X[i])


y_prediction = np.round(y_prediction)

print ('Accuracy from Logistic Regression Human Concatenation: {0}'.format((y_prediction == Y).sum().astype(float) / len(y_prediction)))


##################################### Human Feature Subtraction ##########################################
data = pd.read_csv('Human_Feature_Subtraction.csv',index_col= False)
data = shuffle(data)

msk = np.random.rand(len(data)) < 0.8

train_set = data[msk]

remaining = data[~msk]

val = np.random.rand(len(remaining)) < 0.5

validate_set = remaining[val]

test_set = remaining[~val]

#############################################################################################################
#################### Linear Regression #############################################################

X = np.array(train_set.iloc[:,2:11])   ##### Feature Vector

X0 = np.ones([X.shape[0],1])      ##### Base Feature

X = np.concatenate((X0,X),axis = 1)  ######## Concatenating base feature and Feature Vector 

Y = train_set.iloc[:,11:12].values        ####### Output Vector

theta = np.zeros([1,X.shape[1]])     ########## Corresponding theta Values

learning_rate = 0.03

no_of_iterations = 1000

g,cost = gradientDescent(X,Y,theta,no_of_iterations,learning_rate)

y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(g,X[i])


rms = sqrt(mean_squared_error(Y, y_prediction))
print("Root Mean Square Value of Human Feature Subtraction (Training)")
print(rms)


finalCost = computeCost(X,Y,g)

##################################################Logistic Regression ########################################

weights = logistic_regression(X, Y,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)


y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(weights[1:].T,X[i])


y_prediction = np.round(y_prediction)

print ('Accuracy from Logistic Regression Human_Feature_Subtraction: {0}'.format((y_prediction == Y).sum().astype(float) / len(y_prediction)))

#########################################################################################################################
################################## GSC Feature Concatenation ############################################################

data = pd.read_csv('GSC_Feature_Concatenation.csv',index_col= False)
data = shuffle(data)
print("GSC")

msk = np.random.rand(len(data)) < 0.8

train_set = data[msk]

remaining = data[~msk]

val = np.random.rand(len(remaining)) < 0.5

validate_set = remaining[val]

test_set = remaining[~val]

#############################################################################################################
#################### Linear Regression #############################################################

X = np.array(train_set.iloc[:,2:1024])   ##### Feature Vector

X0 = np.ones([X.shape[0],1])      ##### Base Feature

X = np.concatenate((X0,X),axis = 1)  ######## Concatenating base feature and Feature Vector 

Y = train_set.iloc[:,1024:1025].values        ####### Output Vector

theta = np.zeros([1,X.shape[1]])     ########## Corresponding theta Values


learning_rate = 0.005

no_of_iterations = 10000

g,cost = gradientDescent(X,Y,theta,no_of_iterations,learning_rate)


y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(g,X[i])


rms = sqrt(mean_squared_error(Y, y_prediction))
print("Root Mean Square Value of GSC Feature Concatenation(Training)")
print(rms)


##################################################Logistic Regression ########################################

weights = logistic_regression(X, Y,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)


y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(weights[1:].T,X[i])


y_prediction = np.round(y_prediction)

print ('Accuracy from Logistic Regression GSC Feature Concatenation : {0}'.format((y_prediction == Y).sum().astype(float) / len(y_prediction)))

####################################################################################################################
######################################### GSC Subtraction ###########################################################

data = pd.read_csv('GSC_Feature_Subtraction.csv',index_col= False)
data = shuffle(data)
print("GSC")

msk = np.random.rand(len(data)) < 0.8

train_set = data[msk]

remaining = data[~msk]

val = np.random.rand(len(remaining)) < 0.5

validate_set = remaining[val]

test_set = remaining[~val]

#############################################################################################################
#################### Linear Regression #############################################################

X = np.array(train_set.iloc[:,2:513])   ##### Feature Vector

X0 = np.ones([X.shape[0],1])      ##### Base Feature

X = np.concatenate((X0,X),axis = 1)  ######## Concatenating base feature and Feature Vector 

Y = train_set.iloc[:,513:514].values        ####### Output Vector

theta = np.zeros([1,X.shape[1]])     ########## Corresponding theta Values


learning_rate = 0.005

no_of_iterations = 10000

g,cost = gradientDescent(X,Y,theta,no_of_iterations,learning_rate)

y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(g,X[i])


rms = sqrt(mean_squared_error(Y, y_prediction))
print("Root Mean Square Value of GSC Feature Subtraction (Training)")
print(rms)
finalCost = computeCost(X,Y,g)
##################################################Logistic Regression ########################################

weights = logistic_regression(X, Y,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)


y_prediction = np.zeros(Y.shape)
for i in range(X.shape[1]):
    y_prediction[i][0] = np.dot(weights[1:].T,X[i])


y_prediction = np.round(y_prediction)
print(y_prediction)
print ('Accuracy from Logistic Regression GSC Feature Subtraction: {0}'.format((y_prediction == Y).sum().astype(float) / len(y_prediction)))

