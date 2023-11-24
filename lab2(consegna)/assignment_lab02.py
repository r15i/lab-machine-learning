get_ipython().run_line_magic("matplotlib", "inline")

import typing as tp
import numpy as np
import itertools
from matplotlib import pyplot as plt
import sklearn.metrics as skm
from sklearn.svm import SVC
from sklearn import linear_model


def load_dataset(path: str) -> tp.Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        x, y = data["x"], data["y"]

        x -= x.mean(axis=0)
        x /= x.std(axis=0)

    return x, y


def plot_input(X_matrix: np.ndarray, labels: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.get_cmap("Accent", 4)
    im = ax.scatter(X_matrix[:, 0], X_matrix[:, 1], X_matrix[:, 2], c=labels, cmap=cmap)
    im.set_clim(-0.5, 3.5)
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3], orientation="vertical", cmap=cmap)
    cbar.ax.set_yticklabels(["Sunny", "Rainy", "Cloudy", "Mostly clear"])


def k_split(x: np.ndarray, y: np.ndarray, k: int, shuffle: bool = True) -> tp.Tuple[list[np.ndarray], list[np.ndarray]]:
    if shuffle:
        idx = np.arange(x.shape[0])

        np.random.shuffle(idx)

        x = x[idx]
        y = y[idx]
    

    x_splits = np.array_split(x,k)
    y_splits = np.array_split(y,k)
    
    return x_splits,y_splits 
    
    
def k_fold_cross_validation(x_train: np.ndarray, y_train: np.ndarray, k: int, model: SVC, parameters: dict) -> tp.Tuple[tuple, tuple]:
    x_folds, y_folds = k_split(x_train, y_train, k)

    params = list(itertools.product(*parameters.values()))

    results = {k: 0 for k in params}

    for param in params:
        param = dict(zip(parameters.keys(), param))

        fold_accuracies = []

        results[tuple(param.values())] = round(np.mean(fold_accuracies), 4)

    best_parameters = dict(
        zip(parameters.keys(), params[np.argmax(list(results.values()))])
    )
    best_accuracy = np.max(list(results.values()))
    best = (best_parameters, best_accuracy)

    results = [
        ({k: v for k, v in zip(parameters.keys(), p)}, a) for p, a in results.items()
    ]

    return best, results


"""













ID = None 
np.random.seed(ID)









X, y = load_dataset("data/lux.npz")
print(X.shape, y.shape)









noise = np.random.normal(0, 0.1, X.shape)
X = X + noise












permutation = None 

X = None 
y = None 

m_training = 1000
m_test = 4000

X_train = None 
X_test = None 
y_train = None 
y_test = None 

print("X_train shape:", X_train.shape,"X_test shape:", X_test.shape,"||","y_train shape:",  y_train.shape,"y_test shape:", y_test.shape)

labels, freqs = None 
print("Labels in training dataset: ", labels)
print("Frequencies in training dataset: ", freqs)






plot_input(X_train,y_train)










parameters = {'C': [ 0.01, 0.1, 1, 10]}


svm = None 


best, results = None 

print ('RESULTS FOR LINEAR KERNEL')

print("Best parameter set found:")


print("Score with best parameter:")

print()
print("All scores on the grid:")











parameters = {'C': [0.01, 0.1, 1],'gamma':[0.01,0.1,1.]}


poly2_svm = None 


best, results = None 

print ('RESULTS FOR POLY DEGREE=2 KERNEL')

print("Best parameter set found:")


print("Score with best parameter:")

print()
print("All scores on the grid:")











parameters = {'C': [0.01, 0.1, 1],'gamma':[0.01,0.1, 1]}


degree = 3
poly_svm = None 


best, results = None 

print (f"RESULTS FOR POLY DEGREE={degree} KERNEL")

print("Best parameter set found:")


print("Score with best parameter:")

print()
print("All scores on the grid:")











parameters = {'C': [0.1, 1, 10, 100],'gamma':[0.001, 0.01, 0.1,1]}


rbf_svm = None 


best, results = None 

print ('RESULTS FOR rbf KERNEL')

print("Best parameter set found:")


print("Score with best parameter:")

print()
print("All scores on the grid:")
















best_svm = None 





training_error = None 
test_error = None 

print ("Best SVM training error: %f" % training_error)
print ("Best SVM test error: %f" % test_error)













gamma_values = np.logspace(-5,2,8)
print(gamma_values)





train_acc_list, test_acc_list = [], []









fig, ax = plt.subplots(1,2, figsize=(15,5))

ax[0].plot(gamma_values, train_acc_list)
ax[0].set_xscale('log')
ax[0].set_xlabel('gamma')
ax[0].set_ylabel('Train accuracy')
ax[0].grid(True)

ax[1].plot(gamma_values, test_acc_list)
ax[1].set_xscale('log')
ax[1].set_xlabel('gamma')
ax[1].set_ylabel('Test accuracy')
ax[1].grid(True)

plt.show()












X = X[permutation]
y = y[permutation]

m_training = None 

X_train, X_test = X[:m_training], X[m_training:]
y_train, y_test = y[:m_training], y[m_training:]

labels, freqs = None 
print("Labels in training dataset: ", labels)
print("Frequencies in training dataset: ", freqs)


granularity = 25
x_max = np.abs(X).max()
x_range = np.linspace(-x_max, x_max, granularity)
x_grid = np.stack(np.meshgrid(x_range, x_range, x_range)).reshape(3, -1).T













print ("Best SVM training error: %f" % training_error)
print ("Best SVM test error: %f" % test_error)













rbf_svm_test = None 







rbf_SVM_grid = rbf_svm.predict(x_grid)

rbf_SVM_m = y_test == rbf_svm_test

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection="3d")

ax1.scatter(x_grid[:,0], x_grid[:,1], x_grid[:,2], c=rbf_SVM_grid, linewidth=0, marker="s", alpha=.05,cmap='Accent')

ax1.scatter(X_test[rbf_SVM_m,0], X_test[rbf_SVM_m,1], X_test[rbf_SVM_m,2], c=y_test[rbf_SVM_m], linewidth=.5, edgecolor="k", marker=".",cmap='Accent')
ax1.scatter(X_test[~rbf_SVM_m,0], X_test[~rbf_SVM_m,1], X_test[~rbf_SVM_m,2], c=y_test[~rbf_SVM_m], linewidth=1, edgecolor="r", marker=".",cmap='Accent')
ax1.set_xlim([-x_max, x_max])
ax1.set_ylim([-x_max, x_max])
ax1.set_zlim([-x_max, x_max])

















np.set_printoptions(precision=2, suppress=True) 

u, counts = np.unique(y_test, return_counts=True)
print("Labels and frequencies in test set: ", counts)

confusion_SVM =  
print("\n Confusion matrix SVM  \n \n", confusion_SVM)
print("\n Confusion matrix SVM (normalized)   \n \n", confusion_SVM /counts[:,None] )





fig = plt.figure()
    
im = plt.imshow(confusion_SVM /counts[:,None], cmap="Blues",interpolation='nearest')
plt.xticks([0,1,2,3], ['Sunny', 'Rainy','Cloudy', 'Mostly clear'],ha="right",rotation=30)
plt.yticks([0,1,2,3], ['Sunny', 'Rainy','Cloudy', 'Mostly clear'],ha="right",rotation=30)
cm = confusion_SVM /counts[:,None]
fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
        ha="center", va="center",
        color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
fig.colorbar(im, location='bottom')  
plt.show()







"""
