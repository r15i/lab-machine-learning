import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing
import numpy as np
import matplotlib.pyplot as plt


def plot_line_and_points(w, X, Y):

    print(f"vector {w}")
    print(f"variables")
    print(X)
    print(f"labels{Y}")

    if np.all(w == 0):  # Se il vettore dei pesi  (0, 0, 0)
        plt.scatter(X[:, 1], X[:, 2], c=Y, cmap=plt.cm.Spectral)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    else:
        slope = -w[1] / w[2]
        intercept = -w[0] / w[2]

        plt.scatter(X[:, 1], X[:, 2], c=Y, cmap=plt.cm.Spectral)

        x_vals = np.linspace(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1, 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, 'r--', label='Decision boundary')

        # Aggiungi l'asse sull'origine
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.legend()
    plt.show()


def load_dataset(filename):
    data_train = pd.read_csv(filename)
    data_train = data_train.sample(frac=1).reset_index(drop=True)
    X = data_train.iloc[:, 0:3].values
    Y = data_train.iloc[:, 3].values
    Y = 2*Y-1
    return X, Y


def to_homogeneous(X_training, X_test):

    X_training = np.hstack([np.ones((X_training.shape[0], 1)), X_training])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    return X_training, X_test


def count_errors(current_w, X, Y):

    result = np.dot(X, current_w) * Y
    condition_met = result <= 0

    indices = np.where(condition_met)[0]

    if (len(indices) <= 0):
        return -1, -1
    return len(indices), indices[0]


def perceptron_update(current_w, x, y):

    new_w = current_w + n*x * y
    return new_w


def count_errors(current_w, X, Y):

    result = np.dot(X, current_w) * Y
    condition_met = result <= 0
    indices = np.where(condition_met)[0]

    if (len(indices) <= 0):
        return -1, -1
    r = np.random.randint(0, len(indices))
    return len(indices), indices[0]


def perceptron(X, Y, max_num_iterations):

    num_samples = X.shape[0]
    best_error = num_samples+1

    curr_w = np.zeros(len(X[0]), dtype=int)
    # curr_w = np.array([0, 1, 0.5])
    best_w = curr_w.copy()

    num_misclassified, index_misclassified = count_errors(curr_w, X, Y)
    print("INIZIALIZATION W")
    print(f"best error {best_error} ")
    print(f"index_misclassified {index_misclassified} ")
    # plot_line_and_points(best_w, X, Y)

    if num_misclassified < best_error:
        best_error = num_misclassified
        if (best_error > 0):
            best_w = perceptron_update(
                curr_w, X[index_misclassified], Y[index_misclassified])
    print("BEFORE THE LOOP")
    print(f"best error {best_error} ")
    print(f"index_misclassified {index_misclassified} ")
    plot_line_and_points(best_w, X, Y)
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
        print("")
        print(f"best error {best_error} ")
        print(f"index_misclassified {index_misclassified} ")
        #plot_line_and_points(best_w, X, Y)
    print("")
    print(f"best error {best_error} ")
    print(f"index_misclassified {index_misclassified} ")
    plot_line_and_points(best_w, X, Y)

    if (best_error > 0):
        best_error = best_error/num_samples

    return best_w, best_error


IDnumber = 2122841 + 1
np.random.seed(IDnumber)

n = 1

X = np.array([
    [1, -2, 2],
    [1, 1, 3],
    # [1, 2, 4],
    [1, -3, 5],
    [1, 4, 6],
    [1, 5, 7],
])
Y = np.array([
                -1,
                1,
    #          -1,
               -1,
                1,
                1,
])

best_w = np.array([1, 6, 7.5])  # added from bok


# plot_line_and_points(best_w,X,Y)
# error, index = count_errors(best_w, X, Y)
# best_w = perceptron_update(best_w,X[index],Y[index])
# plot_line_and_points(best_w,X,Y)


res = perceptron(X, Y, 200)

# print(res)
