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
def to_homogeneous(X_training, X_test):
    
    X_training = np.hstack( [np.ones( (X_training.shape[0], 1) ), X_training] )
    X_test = np.hstack( [np.ones( (X_test.shape[0], 1) ), X_test] )
    
    return X_training, X_test


def perceptron_update(current_w, x, y):
    # Place in this function the update rule of the perceptron algorithm
    # Remember that numpy arrays can be treated as generalized variables
    # therefore given array a = [1,2,3,4], the operation b = 10*a will yield
    # b = [10, 20, 30, 40]
    new_w =  current_w + x * y # ADD YOUR CODE HERE  
    return new_w


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


def perceptron(X, Y, max_num_iterations):    
    # Use the previous function as a template to 
    # implement the random version of the perceptron algorithm
    # Initialize some support variables
    num_samples = X.shape[0]
    # best_errors will keep track of the best (minimum) number of errors
    # seen throughout training, used for the update of the best_w variable
    best_error = num_samples+1
     
    # Initialize the weights of the algorith with w=0
    curr_w = np.zeros(len(X[0]) , dtype=int)# ADD YOUR CODE HERE
    # The best_w variable will be used to keep track of the best solution
    best_w = curr_w.copy()

    # compute the number of misclassified samples and the index of the first of them
    num_misclassified, index_misclassified = count_errors(curr_w, X, Y)
    # update the 'best' variables
    if num_misclassified < best_error:
        best_error = num_misclassified# ADD YOUR CODE HERE
        best_w = perceptron_update(curr_w,X[index_misclassified],Y[index_misclassified]) # ADD YOUR CODE HERE
    
    # initialize the number of iterations
    num_iter = 0
    # Main loop continue until all samples correctly classified or max # iterations reached
    # Remember that to signify that no errors were found we set index_misclassified = -1
    while index_misclassified != -1 and num_iter < max_num_iterations:
        
        #update to the current best
        curr_w = best_w
        #use the misclassified index from the previus iteration on the best
        #to correct w
        curr_w = perceptron_update(curr_w, X[index_misclassified], Y[index_misclassified]) 
        #recount the errors  
        num_misclassified, _ = count_errors(curr_w, X, Y)
        
        # Update the best_w if it has fewer errors
        if num_misclassified < best_error:
            best_error = num_misclassified
            best_w = curr_w
        
        num_iter += 1 
        
        # Update the index of the misclassified sample (this time
        # chosen at random one) sample for the next iteration with
        # always the best w found at each iteration
        _, index_misclassified = count_errors(best_w, X, Y) 
        
    #as required, return the best error as a ratio with respect to the total number of samples
    best_error = best_error/num_samples# ADD YOUR CODE HERE
    
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

   
