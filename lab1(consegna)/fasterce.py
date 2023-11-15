import numpy as np
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
    
    return len(indices),indices[0]
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

#inserisco dati dalla slide

current_w = np.array([0, 1, 0.5])  # Vettore dei pesi
X = np.array([
                [1, 1, 1], #giusto 
                [1, -2, 1],#sbagliato
                [1, 2, -2],#sbagliato
            ])  # Matrice delle feature (es. 3 esempi, ogni esempio ha 3 feature)

Y = np.array([
                1,
                1, 
                -1
            ])  # Vettore delle etichette

count_errors(current_w,X,Y)