import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.io import loadmat, savemat
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import glob
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# for file in glob.glob("dataset/*.*"):
for file in glob.glob("datasets/*.*"):
    input = loadmat(file)
    data = input['dataset']

    X = data[:, 0:-1]
    y = data[:, -1]

    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=1000)
    sss.get_n_splits(X, y)
    split_part=0
    name = file.split(".")[0].split("\\")[1]

    for train_index, test_index in sss.split(X, y):
        split_part=split_part+1
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        label_encoder = preprocessing.LabelEncoder()
        y_train_ms = label_encoder.fit_transform(y_train)
        y_test_ms = label_encoder.transform(y_test)

        x_train_1, x_dsel, y_train_1, y_dsel = train_test_split(X_train, y_train_ms, test_size=0.33,random_state=42)
        print(len(x_train_1))
        print(len(X_test))
        print(len(x_dsel))
        print(len(y_train_1))
        print(len(y_test_ms))
        print(len(y_dsel))

        scaler = StandardScaler()
        X_train_ss = scaler.fit_transform(x_train_1)
        X_test_ss = scaler.transform(X_test)
        X_dsel_ss = scaler.transform(x_dsel)

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(X_train_ss)
        imputer = imputer.fit(X_dsel_ss)
        X_train_ms = imputer.transform(X_train_ss)
        X_test_ms = imputer.transform(X_test_ss)
        X_dsel_ms = imputer.transform(X_dsel_ss)

        X_train_processed = X_train_ms
        X_test_processed = X_test_ms
        X_dsel_processed = X_dsel_ms
        y_train_processed = y_train_1
        y_test_processed = y_test_ms
        y_dsel_processed = y_dsel

        processed = [X_train_processed, X_test_processed, y_train_processed, y_test_processed, X_dsel_processed, y_dsel_processed]
        # processed = [X_train_processed, y_train_processed]
        # print(file)
        name = file.split(".")[0].split("\\")[1]
        print(name)

        # How to save data in mat format
        FramStack = np.empty((len(processed),), dtype=object)
        for i in range(len(processed)):
            FramStack[i]=processed[i]


        np.save('processed_8/{}_dir/processed_{}_{}'.format(name, name, split_part), {"FrameStack": FramStack})



