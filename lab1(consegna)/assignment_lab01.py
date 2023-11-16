import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing

IDnumber = 2122841 + 20000
np.random.seed(IDnumber)

def load_dataset(filename):
    data_train = pd.read_csv(filename)
    data_train = data_train.sample(frac=1).reset_index(drop=True)
    X = data_train.iloc[:, 0:3].values
    Y = data_train.iloc[:, 3].values
    Y = 2*Y-1
    return X, Y
X, Y = load_dataset('data/telecom_customer_churn_cleaned.csv')[:10]


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


def to_homogeneous(X_training, X_test):

    X_training = np.hstack([np.ones((X_training.shape[0], 1)), X_training])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    return X_training, X_test


X_training, X_test = to_homogeneous(X_training, X_test)




def count_errors(current_w, X, Y):

    result = np.dot(X, current_w) * Y
    condition_met = result <= 0

    indices = np.where(condition_met)[0]

    if (len(indices) <= 0):
        return -1, -1

    return len(indices), indices[0]


def perceptron_update(current_w, x, y):

    new_w = current_w + x * y
    return new_w


def perceptron_no_randomization(X, Y, max_num_iterations):

    num_samples = X.shape[0]

    best_error = num_samples+1

    curr_w = np.zeros(len(X[0]), dtype=int)

    best_w = curr_w.copy()

    num_misclassified, index_misclassified = count_errors(curr_w, X, Y)

    if num_misclassified < best_error:
        best_error = num_misclassified
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

    best_error = best_error/num_samples

    return best_w, best_error


print(f"seed --->{IDnumber}")
w_found, error = perceptron_no_randomization(X_training, Y_training, 30)
print("Training Error of perceptron_no_randomization (30 iterations): " + str(error))


errors, _ = count_errors(w_found, X_test, Y_test)

true_loss_estimate = errors/len(Y_test)

print("Test Error of perceptron_no_randomization (30 iterations): " +
      str(true_loss_estimate))



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

    best_error = best_error/num_samples

    return best_w, best_error


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


exit()

logreg = linear_model.LogisticRegression(C=1e5)

X_training = X[:m_training]
Y_training = Y[:m_training]

X_test = X[m_training:]
Y_test = Y[m_training:]

logreg.fit(X_training, Y_training)


print("Intercept:", logreg.intercept_)
print("Coefficients:", logreg.coef_)


predicted_training = logreg.predict(X_training)


error_count_training = (predicted_training != Y_training).sum()
error_rate_training = error_count_training/len(Y_training)
print("Error rate on training set: "+str(error_rate_training))


predicted_test = logreg.predict(X_test)


error_count_test = (predicted_test != Y_test).sum()
error_rate_test = error_count_test/len(Y_test)
print("Error rate on test set: " + str(error_rate_test))


feature_names = ["Tenure in Months", "Monthly Charge", "Age"]


idx0 = 0
idx1 = 1

X_reduced = X[:, [idx0, idx1]]


X_training = X_reduced[:m_training]
Y_training = Y[:m_training]

X_test = X_reduced[m_training:]
Y_test = Y[m_training:]


logreg.fit(X_training, Y_training)


predicted_test = logreg.predict(X_test)


error_count_test = (predicted_test != Y_test).sum()


error_rate_test = error_count_test/len(Y_test)
print("Error rate on test set: " + str(error_rate_test))
