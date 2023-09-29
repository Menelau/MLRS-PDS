import numpy as np
from scipy.io import loadmat, savemat
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import glob
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, KFold
from sklearn.preprocessing import StandardScaler

for file in glob.glob("datasets/*.*"):
    print(file)

    input = loadmat(file)
    data = input['dataset']

    X = data[:, 0:-1]
    y = data[:, -1]
    # print(y)

    kf = KFold(n_splits=10)

    # add cross validation for 10 fold
    kf.sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    kf.sss.get_n_splits(X, y)
    split_part=0
    name = file.split(".")[0].split("\\")[1]

    for train_index, test_index in kf.sss.split(X, y):
        split_part=split_part+1
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # standard scaler
        scaler = StandardScaler()
        X_train_ss = scaler.fit_transform(X_train)
        X_test_ss = scaler.transform(X_test)

        # handling the missing data and replace missing values with mean of all the other values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(X_train_ss)
        X_train_ms = imputer.transform(X_train_ss)
        X_test_ms = imputer.transform(X_test_ss)

        # , stratify = y_train
        x_train_f, x_dsel, y_train_f, y_dsel = train_test_split(X_train_ms, y_train, test_size=0.33, stratify=y_train)
        # print(len(x_train_f))
        # print(len(X_test_ms))
        # print(len(x_dsel))
        # print(len(y_train_f))
        # print(len(y_test))
        # print(len(y_dsel))

        label_encoder = preprocessing.LabelEncoder()
        y_train_ms = label_encoder.fit_transform(y_train_f)
        y_test_ms = label_encoder.transform(y_test)
        y_dsel_ms = label_encoder.transform(y_dsel)

        X_train_processed = x_train_f
        X_test_processed = X_test_ms
        X_dsel_processed = x_dsel
        y_train_processed = y_train_ms
        y_test_processed = y_test_ms
        y_dsel_processed = y_dsel_ms

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
