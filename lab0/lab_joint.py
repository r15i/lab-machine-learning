import matplotlib.pyplot as plt
import csv
from scipy import stats
import numpy as np
import sklearn as sl
import sklearn.linear_model as lm


from tqdm import tqdm
import sys
from matplotlib import pyplot as plt




# Load the provided data file with the used car data (you can also have a look at it with any text editor)
filename = "data/km_year_power_price.csv"
test_sample = 10


if(len(sys.argv)>1):
    training_it = int(sys.argv[1])
else:
    training_it = 1
    


lines = csv.reader(open(filename, newline=''), delimiter=',')
dataList = np.array(list(lines))



#array with the column names
col = dataList[0]

#turning into int
#array with the data
data = dataList[1:-test_sample].astype(float)
#test array
test = dataList[-test_sample:].astype(float)



#'km', 'year', 'powerPS', 
X = np.array(data[:,:-1])
#avgPrice

y = np.array(data[:,-1:])

test_X = np.array(test[:,:-1])
test_y = np.array(test[:,-1:])

#constructing model 
model = lm.LinearRegression()
#input data
#CAN BE DONE IN THE INPUT
print(f"Number of iteration : {training_it} ")
model.fit(X,y)

# Fit the model to the data

# Make predictions
prediction = model.predict(X)
error = np.array(np.abs(prediction-y))

print(f"results : " )
print(error)
print(f"everage error ----------:> {np.average(error)}")


#plotting the stuff

plt.plot(X, y, 'o', label='original data')
plt.plot(X, 1 + 0.5*X, 'r', label='fitted line')
plt.legend()
plt.show()
