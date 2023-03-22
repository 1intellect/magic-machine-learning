import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# column names
cols = ["flength", "fwidtth", "fSize", "fConc", "fConc1", "fAsym","fM3long",
        "fM3Trans","fAlpha", "fdist","class"]

#csv =   comma separate values 
df = pd.read_csv("magic04.data", names=cols)

#print the first five values
# print(df.head())

# prints all the rows where the class is labelled 0
# print(df[df["class"]==0])

#To allow the computer to better interpret data we change g and h to 1's and 0s by casting as an int
df["class"] = (df["class"] == "g").astype(int)

# for each of the data points in the last column (labelled class)
for label in cols[:-1]:
    
    plt.figure()
    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7,density=True)
    plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7,density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
   
plt.show()   


#CREATE TRAIN VALIDATION AND TEST DATASETS
#np.split split the data
#df.sample shuffles 100% of the data (frac = 0.5 would be shuffling 50%)
#split the data at 60% - 80% of the data set 
#split the data at 80% - 100% of the data set to be test data
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])   

# we can scale to have all the data with respect to a mean 
def scale_dataset(dataframe):
    X = dataframe[dataframe.cols[:1]].values
    Y = dataframe[dataframe.cols[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    data = np.hstack((X,np.reshape(Y,(-1,1))))

    return data, X, Y






