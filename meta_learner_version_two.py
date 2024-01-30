import pandas as pd
import os
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from pymfe.mfe import MFE

# Upload the data test not the meta feature
meta_features2 = []
for files in glob.glob('s2_test_processed_npy/*'):
    print(files)
    data = np.load(files, allow_pickle=True)

    # Extract all available measures
    mfe = MFE(groups="all")
    mfe.fit([data])
    meta_features_from_file2 = mfe.extract()
    print(meta_features_from_file2)

    print(meta_features_from_file2)
    print(len(meta_features_from_file2))
    print("-----------------test------------------")
    print(meta_features_from_file2[1])
    meta_features2.append(meta_features_from_file2[1])
    f_meta_features2 = [np.nan_to_num(i) for i in meta_features2]  # I replaced nan with 0 specially for knn
    print(f_meta_features2)
    name = files.split(".")[0].split("\\")[1]
    print(name)
    #et the number in the name of the dataset
    string = name
    number = ''.join(filter(str.isdigit, string))
    print(number)
    number = int(number)

test_meta = f_meta_features2

y = []
# df = pd.read_csv('data300.xlsx')
df = pd.read_excel('dataframe_friedman.xlsx')

#esme data ha ro hazf kon vase rank dadan
df1=df.iloc[:,1:]
# df2 = df1.drop(index=0)
print(df1)
# print(df2)
#entekhabe chand column baraye mohayese pool ha dar yek method.  #df1 base hastesh hesam
KNORAE = df1.iloc[:, [1,9,17,25,33,41,49]]
METADES = df1.iloc[:, [2,10,18,26,34,42,50]]
KNORAU = df1.iloc[:, [3,11,19,27,35,43,51]]
DESMI = df1.iloc[:, [4,12,20,28,36,44,52]]
DESP = df1.iloc[:, [5,13,21,29,37,45,53]]
MLA = df1.iloc[:, [6,14,22,30,38,46,54]]
OLA = df1.iloc[:, [7,15,23,31,39,47,55]]
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
print(original_list)
f_y = original_list
# f_y = [[item] for item in original_list]
print(f_y)
print("List size:", len(f_y))
if number in range(len(f_y)):
        del f_y[number-1]
        y = np.array(f_y)
        y = y.reshape(-1)
        print(y)
        print("y  size:", len(y))


def extract_meta_features(folder):
    meta_features_file = os.path.join(folder, "meta_features.npy")
    meta_features = np.load(meta_features_file)
    return meta_features

meta_features = []
for file in glob.glob('mf_sample_train/*'):
    print(file)
    # Load the meta features from the .npy file
    meta_features_from_file = np.load(file, allow_pickle=True)

    # print(meta_features_from_file)
    # print(meta_features_from_file[1]) #in mishe meta feature hesammm
    meta_features.append(meta_features_from_file[1])
    # print(meta_features)
    # print("meta_features:", len(meta_features))
    # meta_features = meta_features[number:]


    list1 = meta_features[:number-1]
    list2 = meta_features[number:]
    mergedList = [*list1, *list2]
    # print(mergedList)
    # print("mergedList  size:", len(mergedList))
    f_meta_features = [np.nan_to_num(i) for i in mergedList] # I replaced nan with 0 specially for knn
    # print(f_meta_features)
    # print("f_meta_features size:", len(f_meta_features))


X = f_meta_features

# Convert to a numpy array
meta_features = np.array(meta_features)

# Create a KNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# y = f_y[number:]
# y = np.array(y)
# y = y.reshape(-1)
# print("y  size:", len(y ))

##to block
# X = f_meta_features
# print(len(f_meta_features))
# X = f_meta_features[number:]
# print("X size:", len(X))

knn.fit(X, y)

test_meta = f_meta_features2

# Use the classifier to predict the label of the test data
predictions = knn.predict(test_meta)

# Print the predicted y label
print("Predicted y label: ", predictions)
