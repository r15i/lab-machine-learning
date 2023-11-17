import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing


np.set_printoptions(linewidth=500, suppress=True)





def load_dataset(filename):
    data_train = pd.read_csv(filename)

    data_train = data_train.sample(frac=1).reset_index(
        drop=True)
    X = data_train.iloc[:, 0:3].values
    Y = data_train.iloc[:, 3].values
    Y = 2*Y-1
    return X, Y

def to_homogeneous(X_training, X_test):

    X_training = np.hstack([np.ones((X_training.shape[0], 1)), X_training])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    return X_training, X_test

IDnumber = 2122841 
np.random.seed(IDnumber)


X, Y = load_dataset('data/telecom_customer_churn_cleaned.csv')


m_training = int(len(X)//(1/0.75))


m_test = len(X) - m_training


X_training = X[:m_training]

Y_training = Y[:m_training]


X_test = X[m_training:]

Y_test = Y[m_training:]

print("Number of samples in the train set:", X_training.shape[0])
print("Number of samples in the test set:", X_test.shape[0])
print("\nNumber of night instances in test:", np.sum(Y_test == -1))
print("Number of day instances in test:", np.sum(Y_test == 1))


scaler = preprocessing.StandardScaler().fit(X_training)


np.set_printoptions(suppress=True)
X_training = scaler.transform(X_training)
print("Mean of the training input data:", X_training.mean(axis=0))
print("Std of the training input data:", X_training.std(axis=0))

X_test = scaler.fit_transform(X_test)
print("Mean of the test input data:", X_test.mean(axis=0))
print("Std of the test input data:", X_test.std(axis=0))

X_training, X_test = to_homogeneous(X_training, X_test)
print("Training set in homogeneous coordinates:")
print(X_training[:10])


def perceptron_update(current_w, x, y):

    new_w = current_w + x * y
    return new_w








def count_errors(current_w, X, Y):

    result = np.dot(X, current_w) * Y
    condition_met = result <= 0

    indices = np.where(condition_met)[0]

    if (len(indices) <= 0):
        return -1, -1

    r = np.random.randint(0, len(indices))
    return len(indices), indices[r]


def perceptron(X, Y, max_num_iterations):

    num_samples = X.shape[0]

    best_error = num_samples+1

    curr_w = np.zeros(len(X[0]), dtype=int)

    best_w = curr_w.copy()

    num_misclassified, index_misclassified = count_errors(curr_w, X, Y)

    if num_misclassified < best_error:
        best_error = num_misclassified
        if (best_error > 0):
            best_w = perceptron_update(
                curr_w, X[index_misclassified], Y[index_misclassified])

    num_iter = 0

    while index_misclassified != -1 and num_iter < max_num_iterations:

        curr_w = best_w
        curr_w = perceptron_update(
            curr_w, X[index_misclassified], Y[index_misclassified])

        num_misclassified, _ = count_errors(curr_w, X, Y)

        if num_misclassified < best_error:
            best_error = num_misclassified
            best_w = curr_w

        num_iter += 1

        _, index_misclassified = count_errors(best_w, X, Y)

    if (best_error < 0):
        return best_w, 0

    best_error = best_error/num_samples
    return best_w, best_error



# Plot the loss with respect to the number of iterations


#seeds
seeds     = IDnumber + np.arange(0,20000,1000) 

#iterations
num_iters = np.arange(3000, 5000, 100)

train_errors_across_seeds = []
for i in seeds:
    train_errors = []
    np.random.seed(i)
    for num_iter in num_iters:
        _, train_error = perceptron(X_training, Y_training, num_iter)
        train_errors.append(train_error)
        
    train_errors_across_seeds.append(train_errors)

plt.figure(figsize=(8, 6))
for train_errors in train_errors_across_seeds:
    plt.plot(num_iters, train_errors)
plt.xlabel('Number of iterations')
plt.ylabel('Training error')
plt.title('Training Error across Seeds')
plt.grid()
plt.legend(['Seed ' + str(seed) for seed in seeds], loc='upper right')
plt.show()
exit()


print(f"SEED {IDnumber} + {seed}")

w_found, error = perceptron(X_training, Y_training, 30)
print("Training Error of perceptron (30 iterations): " + str(error))
errors, _ = count_errors(w_found, X_test, Y_test)
true_loss_estimate = errors/len(Y_test)
print("Test Error of perceptron (30 iterations): " + str(true_loss_estimate))

w_found, error = perceptron(X_training, Y_training, 3000)
print("Training Error of perceptron (3000 iterations): " + str(error))
num_errors, _ = count_errors(w_found, X_test, Y_test)
true_loss_estimate = num_errors/len(Y_test)
print("Test Error of perceptron (3000 iterations): " + str(true_loss_estimate))

