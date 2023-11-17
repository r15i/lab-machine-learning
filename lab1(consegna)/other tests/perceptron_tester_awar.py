import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing
import tqdm

def load_dataset(filename):
    data_train = pd.read_csv(filename)
    data_train = data_train.sample(frac=1).reset_index(drop=True) 
    X = data_train.iloc[:, 0:3].values 
    Y = data_train.iloc[:, 3].values 
    Y = 2*Y-1 
    return X,Y



def count_errors(current_w, X, Y):
    # This function:
    # computes the number of misclassified samples
    # returns the index of all misclassified samples
    # if there are no misclassified samples, returns -1 as index
    
    # ADD YOUR CODE HERE
    # WRITE THE FUNCATION
    
    result = np.dot(X, current_w) * Y
    condition_met = result <= 0
    
    #indices that meet the condition 
    indices = np.where(condition_met)[0]
    
    if(len(indices)<=0):
        return -1,-1
    
    #random index
    r = np.random.randint(0, len(indices))  
    return len(indices),indices[r]

def to_homogeneous(X_training, X_test):
    
    X_training = np.hstack( [np.ones( (X_training.shape[0], 1) ), X_training] )
    X_test = np.hstack( [np.ones( (X_test.shape[0], 1) ), X_test] )
    
    return X_training, X_test

#update rule used in the iterative algorithm
def perceptron_update(current_w, x, y):
    
    new_w=current_w+float(y)*x
    return new_w

def perceptron(X, Y, max_num_iterations):
    
    dim = int(Y.shape[0])
    curr_w = np.zeros(X.shape[1])
    best_w = np.zeros(X.shape[1])
    num_samples = dim
    best_error = 10000000000 
    
    
#      index_misclassified = -1 #will be ovewritten
#     num_misclassified = 0 #will be ovewritten
    
    
    index_misclassified = -2 
       #main loop continue until all samples correctly classified or max # iterations reached
 
    
    num_iter = 1
    
    while ((index_misclassified != -1) and (num_iter < max_num_iterations)):
        
        num_misclassified = 0 #will be ovewritten
        permutation=np.random.permutation(num_samples)
        X=X[permutation]
        Y=Y[permutation]
        
        for i in range(num_samples):
            if (np.inner(curr_w,X[i,:])*Y[i]<=0): 
                index_misclassified = i
                break
         #update  error count, keep track of best solution
            
        curr_w=perceptron_update(curr_w,X[index_misclassified],Y[index_misclassified]) #update the current solution                  

        for i in range(num_samples):
            if (np.inner(curr_w,X[i,:])*Y[i]<=0):
                num_misclassified+=1
                
       
        if num_misclassified == 0:
            index_misclassified = -1
            
        curr_error=num_misclassified/num_samples
        
        if curr_error<best_error:
            best_error=curr_error
            best_w=curr_w
            
        num_iter += 1
        
    
    return best_w, best_error


X, Y = load_dataset('data/telecom_customer_churn_cleaned.csv')


m_training = int(len(X)//(1/0.75))
m_test =  len(X) - m_training
X_training =  X[:m_training]
Y_training =  Y[:m_training]
X_test =   X[m_training:] 
Y_test =  Y[m_training:]  

scaler = preprocessing.StandardScaler().fit(X_training) 
np.set_printoptions(suppress=True) 
X_training = scaler.transform(X_training) 
X_test = scaler.fit_transform(X_test) 

X_training, X_test = to_homogeneous(X_training, X_test)

max_iterations_values=[30,3000]
#max_iterations_values=[30]
seeds = [1,250,20000]
#seeds = [1]
IDnumber = 2122841    

for j in seeds:
    seed = IDnumber + j
    print(f"preceptron with seed 2122841 + {j} = {seed}\n")
    for i in max_iterations_values:
        np.random.seed(seed)
        max_iterations = i  
        #training
        w_found, train_error = perceptron(X_training,Y_training, i)
        print(f"Training Error of perceptron ({i}\t\titerations): \t{train_error} " )
        
        #train errors
        #train_error, _ = count_errors(w_found, X_training,Y_training)
        #train_error_perceptron = train_error/len(Y_training)
        #test errors
        test_error, _ = count_errors(w_found, X_test,Y_test)        
        test_error_perceptron = test_error/len(Y_test)
        #print(f"Training Error of perceptron ({i}\t\titerations): \t{train_error_perceptron} " )
        print(f"Test     Error of perceptron ({i}\t\titerations): \t{test_error_perceptron}\n" )

   
