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

test_sample = 0

lines = csv.reader(open(filename, newline=''), delimiter=',')
dataList = np.array(list(lines))
#array with the column names
col = dataList[0]
#turning into int
#array with the data
data = dataList[1:].astype(float)

#'km', 'year', 'powerPS', 
#matrix input
X = np.array(data[:,:-1])

#splitted stuff
km = np.array(X[:,0])
year = np.array(X[:,1])
powerPS = np.array(X[:,2])


#avgPrice
y = np.array(data[:,-1])


#scipy model 

#single models for single features
scipyModelCorr = []

for i in range(0,3):
    scipyModelCorr.append(stats.linregress(X[:,i],y))






#find the max one and make a prediction 
    
"""
plt.plot(X, y, 'o', label='original data')
plt.plot(X, 1 + 0.5*X, 'r', label='fitted line')
plt.legend()
plt.show()
"""