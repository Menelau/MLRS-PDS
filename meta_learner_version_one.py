import pandas as pd
import os
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pymfe.mfe import MFE
import re

y = []
# df = pd.read_csv('data300.xlsx')
df = pd.read_excel('dataframe_friedman.xlsx')

#esme data ha ro hazf kon vase rank dadan
df1=df.iloc[:,1:]
df2 = df1.drop(index=0)
# print(df1)
print(df2)
#entekhabe chand column baraye mohayese pool ha dar yek method.  #df1 base hastesh hesam
KNORAE = df2.iloc[:, [1,9,17,25,33,41,49]]
METADES = df2.iloc[:, [2,10,18,26,34,42,50]]
KNORAU = df2.iloc[:, [3,11,19,27,35,43,51]]
DESMI = df2.iloc[:, [4,12,20,28,36,44,52]]
DESP = df2.iloc[:, [5,13,21,29,37,45,53]]
MLA = df2.iloc[:, [6,14,22,30,38,46,54]]
OLA = df2.iloc[:, [7,15,23,31,39,47,55]]
algorithm = KNORAE
# algorithm = METADES
# algorithm = KNORAU
# algorithm = DESMI
# algorithm = DESP
# algorithm = MLA
# algorithm = OLA
rank_knorae = algorithm.rank(ascending=False, method='min',axis=1)
print(algorithm)
highest_value_col_name = algorithm.idxmax(axis=1)
print(highest_value_col_name)
original_list = highest_value_col_name
f_y = [[item] for item in original_list]
print(f_y)
print("List size:", len(highest_value_col_name))

def extract_meta_features(folder):
    meta_features_file = os.path.join(folder, "meta_features.npy")
    meta_features = np.load(meta_features_file)
    return meta_features

meta_features = []
for file in glob.glob('mf_sample_train/*'):
    print(file)
    # Load the meta features from the .npy file
    meta_features_from_file = np.load(file, allow_pickle=True)

    print(meta_features_from_file)
    print(meta_features_from_file[1]) #in mishe meta feature hesammm
    meta_features.append(meta_features_from_file[1])
    f_meta_features = [np.nan_to_num(i) for i in meta_features] # I replaced nan with 0 specially for knn
    print(f_meta_features)  
    print("List size:", len(f_meta_features))

# Convert to a numpy array
meta_features = np.array(meta_features)

# Create a KNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

X = f_meta_features
y = f_y
y = np.array(y)
y = y.reshape(-1)

knn.fit(X, y)


# Upload the test data
meta_features2 = []
for files in glob.glob('www/*'):
    print(files)
    meta_features_from_file2 = np.load(files, allow_pickle=True)

    print(meta_features_from_file2)
    print("-------------------------------------------------------------------------------------------------")
    print(meta_features_from_file2[1])  
    meta_features2.append(meta_features_from_file2[1])
    f_meta_features2 = [np.nan_to_num(i) for i in meta_features2]  # I replaced nan with 0 specially for knn
    print(f_meta_features2)

test_meta = f_meta_features2

# Use the classifier to predict the label of the test data
predictions = knn.predict(test_meta)

# Print the predicted y label
print("Predicted y label: ", predictions)
