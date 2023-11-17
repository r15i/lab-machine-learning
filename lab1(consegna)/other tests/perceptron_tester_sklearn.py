from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np


def load_dataset(filename):
    data_train = pd.read_csv(filename)
    #permute the data
    data_train = data_train.sample(frac=1).reset_index(drop=True) # shuffle the data
    X = data_train.iloc[:, 0:3].values # Get first two columns as the input
    Y = data_train.iloc[:, 3].values # Get the third column as the label
    Y = 2*Y-1 # Make sure labels are -1 or 1 (0 --> -1, 1 --> 1)
    return X,Y
def to_homogeneous(X_training, X_test):
    # Add a 1 to each sample (homogeneous coordinates)
    X_training = np.hstack( [np.ones( (X_training.shape[0], 1) ), X_training] )
    X_test = np.hstack( [np.ones( (X_test.shape[0], 1) ), X_test] )
    
    return X_training, X_test

# Load the dataset
X, Y = load_dataset('data/telecom_customer_churn_cleaned.csv')


m_training = int(len(X)//(1/0.75))

# Dividi il dataset in training set e test set
# m_test is the number of samples in the test set (total-training)
m_test =  len(X) - m_training

# X_training = instances for training set
X_training =  X[:m_training]
# Y_training = labels for the training set
Y_training =  Y[:m_training]# ADD YOUR CODE HERE

# X_test = instances for test set
X_test =   X[m_training:] # ADD YOUR CODE HERE
# Y_test = labels for the test set
Y_test =  Y[m_training:]  # ADD YOUR CODE HERE

# print("Number of samples in the train set:", X_training.shape[0])
# print("Number of samples in the test set:", X_test.shape[0])
# print("\nNumber of night instances in test:", np.sum(Y_test==-1))
# print("Number of day instances in test:", np.sum(Y_test==1))

# standardize the input matrix
# the transformation is computed on training data and then used on all the 3 sets
scaler = preprocessing.StandardScaler().fit(X_training) 

np.set_printoptions(suppress=True) # sets to zero floating point numbers < min_float_eps
X_training = scaler.transform(X_training) # ADD YOUR CODE HERE
#print ("Mean of the training input data:", X_training.mean(axis=0))
#print ("Std of the training input data:",X_training.std(axis=0))

X_test = scaler.fit_transform(X_test) # ADD YOUR CODE HERE
#print ("Mean of the test input data:", X_test.mean(axis=0))
#print ("Std of the test input data:", X_test.std(axis=0))

max_iterations_values=[30,3000,30000]

seeds = [1,250,20000]
IDnumber = 2122841    #setting this allows to set the same random

#np.random.seed(0)

for j in seeds:
    seed = IDnumber + j
    print(f"preceptron_sklearn with seed 2122841 + {j} = {seed}\n")
    for i in max_iterations_values:
        # Addestra il perceptron di sklearn specificando il numero di iterazioni
        max_iterations = i  # Specifica il numero massimo di iterazioni
        sklearn_perceptron = Perceptron(max_iter=max_iterations, random_state=seed)
        sklearn_perceptron.fit(X_training, Y_training)
        # Valuta l'errore sui dati di training
        train_error_sklearn = 1 - sklearn_perceptron.score(X_training, Y_training)
        # Valuta l'errore sui dati di test
        test_error_sklearn = 1 - sklearn_perceptron.score(X_test, Y_test)
        print(f"Training Error of perceptron_sklearn ({i}\t\titerations): \t{train_error_sklearn} " )
        print(f"Test     Error of perceptron_sklearn ({i}\t\titerations): \t{test_error_sklearn}\n" )

   
