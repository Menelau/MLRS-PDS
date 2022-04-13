import glob
import deslib
from deslib.static import Oracle
from sklearn.utils.validation import check_X_y, check_array
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
import numpy as np
from sklearn.metrics import check_scoring
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from deslib.static.single_best import SingleBest



dict_final_results_oracle = {}
dict_standard_deviation = {}


for files in glob.glob('processed_888_ino/*'):
    print(files)

    oracle_results = []
    sd_oracle = []

    for items in glob.glob("{}/*.*".format(files)):
        # print(items)




        file_name = files.split("\\")[1]
        # print(file_name)
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        print(item_name)
        ss = file_name.split("_")[0]
        # model_load = glob.glob("model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("model_files/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
        model_load = glob.glob("Models_FLT_f/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]
        # print(model_load)
        with open(model_load, 'rb') as file:
            my_model = pickle.load(file)

            data = np.load(items, allow_pickle=True)
            # print(data)
            input_list = data.tolist()  # How to get x_train
            x_train = input_list['FrameStack'][0]
            x_test = input_list['FrameStack'][1]
            y_train = input_list['FrameStack'][2]
            y_test = input_list['FrameStack'][3]
            x_dsel = input_list['FrameStack'][4]
            y_dsel = input_list['FrameStack'][5]
            # print(x_dsel)
            # print(y_dsel)
            # print(len(x_train))
            # print(len(x_test))

            x_train = x_train[: , :18]
            x_dsel = x_dsel[: , :18]
            x_test = x_test[: , :18]

            # x_trainnn, x_dsel, y_trainnn, y_dsel = train_test_split(x_train, y_train, test_size=0.33,random_state=42)
            # print(len(x_trainnn))
            # print(len(x_dsel))
            # print(len(y_trainnn))
            # print(len(y_dsel))

            pool_classifiers = my_model
            oracleeeee = deslib.static.oracle.Oracle(pool_classifiers=my_model, random_state=None,
                                                     n_jobs=-1)
            oracleeeee.fit(x_dsel, y_dsel)
            y_pred = oracleeeee.predict(x_test, y_test)
            accuracyyy = accuracy_score(y_test, y_pred)
            # print(accuracyyy)
            oracle_results.append(accuracyyy)

        np.std(oracle_results, dtype=np.float64)
        sd_oracle.append(np.std(oracle_results, dtype=np.float64))

    print(oracle_results)
    mean_sb= mean(oracle_results)
    print(mean_sb)
    print(sd_oracle)
    mean_sd_sb = mean(sd_oracle)
    print(mean_sd_sb)

    dict_final_results_oracle["{}".format(files)] = '{:.4f}'.format(mean(oracle_results))
    print(dict_final_results_oracle)
    #     save dictionarty
    # make a list of the methods and do in a  code

    np.std(oracle_results, dtype=np.float64)
    sd_oracle.append(np.std(oracle_results, dtype=np.float64))
    dict_standard_deviation["{}".format(files)] = '{:.4f}'.format(mean(sd_oracle))
    print(dict_final_results_oracle.values())

print(dict_standard_deviation.values())
standard_deviation = (dict_standard_deviation.values())

    #
    #         sb_bdt = SingleBest(pool_classifiers = pool_classifiers, scoring = None, random_state = None, n_jobs = -1)
    #         sb_bdt.fit(x_dsel, y_dsel)
    #         sb_bdt.predict(x_dsel)
    #         acc = sb_bdt.score(x_dsel, y_dsel, sample_weight=None)
    #         single_best_results.append(acc)
    #
    #     np.std(single_best_results, dtype=np.float64)
    #     sd_single_best.append(np.std(single_best_results, dtype=np.float64))
    #
    #
    # print(single_best_results)
    # mean_sb= mean(single_best_results)
    # print(mean_sb)
    # print(sd_single_best)
    # mean_sd_sb = mean(sd_single_best)
    # print(mean_sd_sb)



