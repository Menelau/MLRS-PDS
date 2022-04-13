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
from sklearn.metrics import average_precision_score

dict_sb_results = {}
dict_standard_deviation = {}


for files in glob.glob('processed_888_ino/*'):
    print(files)

    single_best_results = []
    sd_single_best = []

    for items in glob.glob("{}/*.*".format(files)):
        # print(items)

        standard_deviation = []

        accuracy = []


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

            # x_train = x_train[: , :9]
            # x_dsel = x_dsel[: , :9]
            # x_test = x_test[: , :9]
            # x_trainnn, x_dsel, y_trainnn, y_dsel = train_test_split(x_train, y_train, test_size=0.33,random_state=42)
            # print(len(x_trainnn))
            # print(len(x_dsel))
            # print(len(y_trainnn))
            # print(len(y_dsel))

            pool_classifiers = my_model

            b_list = ["German",
                "Haberman",
                "Heart",
                "ILPD",
                "Ionosphere",
                "Laryngeal1",
                "Lithuanian",
                "Liver",
                "Magic",
                "Mammographic",
                "Monk"
                "Phoneme",
                "Pima",
                "Sonar",
                "Vertebral",
                "Adult",
                "Banana",
                "Blood",
                "Breast",
                "Weaning"]

            if files in b_list:
                sb_bdt = SingleBest(pool_classifiers=pool_classifiers, scoring="recall", random_state=None, n_jobs=-1)



                # "CTG",
                # "Ecoli",
                # "Faults",
                # "Laryngeal3",
                # "GLASS",
                # "Segmentation",
                # "Thyroid",
                # "Wine"
                # "Vehicle",
                # "WDVG",



            else :
                sb_bdt = SingleBest(pool_classifiers=pool_classifiers, scoring="f1_macro", random_state=None, n_jobs=-1)
            # accuracy_score
            sb_bdt.fit(x_dsel, y_dsel)
            sb_bdt.predict(x_dsel)
            acc = sb_bdt.score(x_test, y_test, sample_weight=None)
            single_best_results.append(acc)




        np.std(single_best_results, dtype=np.float64)
        sd_single_best.append(np.std(single_best_results, dtype=np.float64))

    # ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
    #  'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
    #  'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples',
    #  'jaccard_weighted', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error',
    #  'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance',
    #  'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error',
    #  'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
    #  'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
    #  'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted',
    #  'top_k_accuracy', 'v_measure_score']
    print(single_best_results)
    mean_sb= mean(single_best_results)
    print(mean_sb)
    print(sd_single_best)
    mean_sd_sb = mean(sd_single_best)
    print(mean_sd_sb)

    dict_sb_results["{}".format(files)] = '{:.4f}'.format(mean(single_best_results))
    print(dict_sb_results)
    #     save dictionarty
    # make a list of the methods and do in a  code

    np.std(single_best_results, dtype=np.float64)
    sd_single_best.append(np.std(single_best_results, dtype=np.float64))
    dict_standard_deviation["{}".format(files)] = '{:.4f}'.format(mean(sd_single_best))
    print(dict_sb_results.values())

print(dict_standard_deviation.values())
standard_deviation = (dict_standard_deviation.values())

# import glob
# import deslib
# from deslib.static import Oracle
# from sklearn.utils.validation import check_X_y, check_array
# import self
# from numpy import mean
# import numpy as np
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import AdaBoostClassifier
# import pandas as pd
# import pickle
# from sklearn.linear_model import Perceptron
# from tabulate import tabulate
# from dataclasses import make_dataclass
# from sklearn.ensemble import RandomForestClassifier
# from deslib.static.base import BaseStaticEnsemble
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn.metrics import check_scoring
# from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
# from deslib.static.single_best import SingleBest
#
#
#
# dict_sb_results = {}
# dict_standard_deviation = {}
#
#
# for files in glob.glob('processed_6/*'):
#     print(files)
#
#     single_best_results = []
#     sd_single_best = []
#
#     for items in glob.glob("{}/*.*".format(files)):
#         # print(items)
#
#
#
#
#         file_name = files.split("\\")[1]
#         # print(file_name)
#         item_name = items.split("\\")[2].split(".")[0].split("_")[2]
#         print(item_name)
#         ss = file_name.split("_")[0]
#         # model_load = glob.glob("model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("model_files/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("model_files/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
#         model_load = glob.glob("model_files/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("Models_FLT_10_2/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]
#         # print(model_load)
#         with open(model_load, 'rb') as file:
#             my_model = pickle.load(file)
#
#             data = np.load(items, allow_pickle=True)
#             # print(data)
#             input_list = data.tolist()  # How to get x_train
#             x_train = input_list['FrameStack'][0]
#             x_test = input_list['FrameStack'][1]
#             y_train = input_list['FrameStack'][2]
#             y_test = input_list['FrameStack'][3]
#             x_dsel = input_list['FrameStack'][4]
#             y_dsel = input_list['FrameStack'][5]
#             # print(x_dsel)
#             # print(y_dsel)
#             # print(len(x_train))
#             # print(len(x_test))
#
#
#             # x_trainnn, x_dsel, y_trainnn, y_dsel = train_test_split(x_train, y_train, test_size=0.33,random_state=42)
#             # print(len(x_trainnn))
#             # print(len(x_dsel))
#             # print(len(y_trainnn))
#             # print(len(y_dsel))
#
#             pool_classifiers = my_model
#
#             sb_bdt = SingleBest(pool_classifiers = pool_classifiers, scoring = None, random_state = None, n_jobs = -1)
#             sb_bdt.fit(x_dsel, y_dsel)
#             sb_bdt.predict(x_dsel)
#             acc = sb_bdt.score(x_test, y_test, sample_weight=None)
#             single_best_results.append(acc)
#
#         np.std(single_best_results, dtype=np.float64)
#         sd_single_best.append(np.std(single_best_results, dtype=np.float64))
#
#
#     print(single_best_results)
#     mean_sb= mean(single_best_results)
#     print(mean_sb)
#     print(sd_single_best)
#     mean_sd_sb = mean(sd_single_best)
#     print(mean_sd_sb)
#
#     dict_sb_results["{}".format(files)] = '{:.4f}'.format(mean(single_best_results))
#     print(dict_sb_results)
#     #     save dictionarty
#     # make a list of the methods and do in a  code
#
#     np.std(single_best_results, dtype=np.float64)
#     sd_single_best.append(np.std(single_best_results, dtype=np.float64))
#     dict_standard_deviation["{}".format(files)] = '{:.4f}'.format(mean(sd_single_best))
#     print(dict_sb_results.values())
#
# print(dict_standard_deviation.values())
# standard_deviation = (dict_standard_deviation.values())

            # meannn = mean(acc)
            # print(meannn)
    #
    # dict_final_results["{}".format(files)] = mean(ola_results)
    # print(dict_final_results)



# sb_bdt = SingleBest(pool_classifiers=bagging_with_decision_tree).fit(x_dsel, y_dsel)
# print(sb_bdt)


