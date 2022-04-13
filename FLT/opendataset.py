import glob
from deslib.des import KNORAE
from deslib.static import Oracle
from scipy.io import loadmat
from sklearn.utils.validation import check_X_y, check_array, check_random_state
import self
from numpy import mean
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import pickle
from sklearn.linear_model import Perceptron
from tabulate import tabulate
from dataclasses import make_dataclass
from sklearn.ensemble import RandomForestClassifier
from deslib.static.base import BaseStaticEnsemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.io import arff
import scipy
from scipy import io
import os
from scipy.io import loadmat
import numpy
from numpy import asarray, size
from numpy import savetxt
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat, savemat
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import glob
from numpy import genfromtxt
import csv
import arff
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat, savemat
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.io import arff
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import ShuffleSplit
from scipy.io import arff
import csv
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from deslib.des.knora_e import KNORAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import glob
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from scipy.io import arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import ClassifierMixin
from deslib.dcs.ola import OLA
from deslib.des import KNOP
from deslib.des import KNORAU




knorae_results = []
accuracy = []
dict_final_results = {}
dict_standard_deviation = {}

for files in glob.glob('preprocessed_files/*'):
    print(files)

    decisionTreeAccutrcies = []
    standard_deviation_decisionTreeAccutrcies = []


    for items in glob.glob("{}/*.*".format(files)):
        print(items)

        standard_deviation = []

        file_name = files.split("\\")[1]
        print(file_name)
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        print(item_name)
        # model_load = glob.glob("model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/glasst9.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/glasst7.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/glasst201.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/glasst92.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("glasst203.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/model203.pkl".format(file_name, item_name))[0]
        model_load = glob.glob("model203.pkl".format(file_name, item_name))[0]
        print(model_load)
        with open(model_load, 'rb') as file:
            my_model = pickle.load(file)
            # print('horaaaa')
            # type(my_model)
            #
            # print(my_model)
            # print(type(my_model))
            # ist = my_model[0]

            # print(ist)



            # entries = os.listdir()
            data = np.load(items, allow_pickle=True)
            # print(data)
            input_list = data.tolist()  # How to get x_train
            x_train = input_list['FrameStack'][0]
            # print(x_train)
            # FramStack = np.empty((len(x_train),), dtype=object)
            # for i in range(len(x_train)):
            #     FramStack[i] = x_train[i]
            # savemat("myfile.mat", {"FrameStack": FramStack})
                # np.save('{}_dir/processed_{}_{}'.format(name, name, split_part), {"FrameStack": FramStack})
            x_test = input_list['FrameStack'][1]
            y_train = input_list['FrameStack'][2]
            y_test = input_list['FrameStack'][3]
            # print(x_train)
            # print(len(x_train))
            # print(len(x_test))

            x_trainnn, x_dsel, y_trainnn, y_dsel = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
            # print(len(x_trainnn))
            # print(len(x_dsel))
            # print(len(y_trainnn))
            # print(len(y_dsel))




            pool_classifiers = my_model
            # pool_classifiers = ist


            # pool_classifiers.fit(x_train, y_train)
            # pool_classifiers.fit(x_train, y_train)
            # selected_feat = x_train.columns[(sel.get_support())]
            # pool_classifiers.fit(x_trainnn, y_trainnn)

            # Initialize the DES model
            # knorae = KNORAE(pool_classifiers)
            #OlA model
            # knorae = OLA(pool_classifiers)
            knorae = KNORAU(pool_classifiers)

            # Preprocess the Dynamic Selection dataset (DSEL)
            knorae.fit(x_dsel, y_dsel)

            # Predict new examples:
            yh = knorae.predict(x_test)
            acc = accuracy_score(y_test, yh)
            print(acc)
            knorae_results.append(acc)

    dict_final_results["{}".format(files)] = mean(knorae_results)
    print(dict_final_results)
    np.std(knorae_results, dtype=np.float64)
    standard_deviation.append(np.std(knorae_results, dtype=np.float64))
    dict_standard_deviation["{}".format(files)] = mean(standard_deviation)
    print(dict_standard_deviation.values())

print(dict_standard_deviation.values())
standard_deviation = (dict_standard_deviation.values())




