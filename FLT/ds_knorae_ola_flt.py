import deslib
from deslib.dcs import OLA
from deslib.des.knora_e import KNORAE
from deslib.static import Oracle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import glob
import numpy as np
from numpy import mean
import pickle
import deslib
from deslib.dcs import OLA, MLA
from deslib.des import METADES, KNORAU, DESMI
from deslib.des.knora_e import KNORAE
from sklearn.metrics import accuracy_score
import glob
import numpy as np
from numpy import mean
import pickle

dict_final_results = {}
dict_standard_deviation = {}
for files in glob.glob('processed_8/*'):
    print(files)
    standard_deviation = []
    knorae_results = []
    accuracy = []
    kk = []

    for items in glob.glob("{}/*.*".format(files)):
        print(items)

        file_name = files.split("\\")[1]
        # print(file_name)
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        # print(item_name)
        ss = file_name.split("_")[0]
        # print(ss)
        # model_load = glob.glob("models_f/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
        model_load = glob.glob("Models_FLT_f/{}/{}_{}.pkl".format(file_name,ss, item_name))[0]
        # model_load = glob.glob("model_calibre/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]

        # print(model_load)
        # print(type(model_load))

        with open(model_load, 'rb') as file:
            # my_model = pickle.load(file)
            mm = pickle.load(file)
            my_model = kk.append(mm)

            # print(type(my_model))
            data = np.load(items, allow_pickle=True)
            # print(data)
            input_list = data.tolist()  # How to get x_train
            x_train = input_list['FrameStack'][0]
            x_test = input_list['FrameStack'][1]
            y_train = input_list['FrameStack'][2]
            y_test = input_list['FrameStack'][3]
            x_dsel = input_list['FrameStack'][4]
            y_dsel = input_list['FrameStack'][5]
            # print(x_train)
            # print(type(x_train))
            # print(len(x_train))
            # print(len(x_test))
            # print(len(x_dsel))

            # x_train = x_train[: , :18]
            # x_dsel = x_dsel[: , :18]
            # x_test = x_test[: , :18]

            # x_merge = np.concatenate((x_train, x_dsel))
            # y_merge = np.concatenate((y_train, y_dsel))

            # x_merge = x_train.join(x_dsel)
            # y_merge = y_train.join(y_dsel)
            x_merge = np.append(x_train, x_dsel, axis=0)
            y_merge = np.append(y_train, y_dsel, axis=0)

            pool_classifiers = my_model
            method = KNORAE
            # method = MLA
            # method = OLA
            # method = METADES
            # method = KNORAU
            # method = DESMI

            ensemble = method(pool_classifiers)
            # ensemble.fit(x_train, y_train)
            ensemble.fit(x_merge, y_merge)
            # Predict new examples:
            yh = ensemble.predict(x_test)
            acc = accuracy_score(y_test, yh)
            print(acc)
            knorae_results.append(acc)

    dict_final_results["{}".format(files)] = '{:.4f}'.format(mean(knorae_results))
    print(dict_final_results)
    np.std(knorae_results, dtype=np.float64)
    standard_deviation.append(np.std(knorae_results, dtype=np.float64))
    dict_standard_deviation["{}".format(files)] = '{:.4f}'.format(mean(standard_deviation))

print(dict_final_results.values())
print(dict_standard_deviation.values())












# dict_final_results = {}
# dict_standard_deviation = {}
#
# for files in glob.glob('processed_8/*'):
#     print(files)
#     standard_deviation = []
#     knorae_results = []
#     accuracy = []
#
#     for items in glob.glob("{}/*.*".format(files)):
#         print(items)
#
#         file_name = files.split("\\")[1]
#         # print(file_name)
#         item_name = items.split("\\")[2].split(".")[0].split("_")[2]
#         # print(item_name)
#         ss = file_name.split("_")[0]
#         # print(ss)
#         # model_load = glob.glob("model_files/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("model_files/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("model_files/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("model_files/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("model_files/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
#         # model_load = glob.glob("Models_FLT_10_2/{}/{}_{}.pkl".format(file_name,ss, item_name))[0]
#         # model_load = glob.glob("Models_FLT/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]
#         # model_load = glob.glob("Models_FLT_f/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]
#         model_load = glob.glob("processed_temp/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]
#
#
#         with open(model_load, 'rb') as file:
#             my_model = pickle.load(file)
#             # print(type(my_model))
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
#             # print(x_train)
#             print(len(x_train))
#             print(len(x_test))
#             print(len(x_dsel))
#
#             #vase dt feature issue
#             # x_train = x_train[: , :18]
#             # x_dsel = x_dsel[: , :18]
#             # x_test = x_test[: , :18]
#             import pandas as pd
#             df = pd.DataFrame(data=x_train)
#             # print(df)
#
#             x_merge = np.concatenate((x_train, x_dsel), axis=0)
#             y_merge = np.concatenate((y_train, y_dsel), axis=0)
#             print(len(x_merge))
#
#             pool_classifiers = my_model
#             # Initialize the DES model
#             # knorae = KNORAE(pool_classifiers)
#             # # merg data train and dsel
#             # # Preprocess the Dynamic Selection dataset (DSEL)
#             # knorae.fit(x_merge, y_merge)
#             # # Predict new examples:
#             # yh = knorae.predict(x_test)
#
#             method = OLA
#             # method = KNORAE
#
#             ensemble = method(pool_classifiers)
#
#             # Preprocess the Dynamic Selection dataset (DSEL)
#             ensemble.fit(x_merge, y_merge)
#             # ensemble.fit(x_train, y_train)
#
#             # Predict new examples:
#             yh = ensemble.predict(x_test)
#
#             acc = accuracy_score(y_test, yh)
#             print(acc)
#             knorae_results.append(acc)
#
#             # meannn = mean(acc)
#             # print(mean(ola_results))
#
#     dict_final_results["{}".format(files)] = '{:.4f}'.format(mean(knorae_results))
#     print(dict_final_results)
# #     save dictionarty
# # make a list of the methods and do in a  code
#
#     dict_final_results["{}".format(files)] = '{:.4f}'.format(mean(knorae_results))
#     print(dict_final_results)
#     np.std(knorae_results, dtype=np.float64)
#     standard_deviation.append(np.std(knorae_results, dtype=np.float64))
#     dict_standard_deviation["{}".format(files)] = '{:.4f}'.format(mean(standard_deviation))
#     print(dict_final_results.values())
#
#
# print(dict_standard_deviation.values())
# standard_deviation = (dict_standard_deviation.values())

# import pandas as pd
# df1 = pd.DataFrame(x_train)
# df2 = pd.DataFrame(x_dsel)
# frames = [df1, df2]
#
# x_merge_pd = pd.concat(frames)
# x_merge = x_merge_pd.to_numpy()
# # print(len(x_merge))
# df3 = pd.DataFrame(y_train)
# df4 = pd.DataFrame(y_dsel)
# frames = [df3, df4]
#
# y_merge_pd = pd.concat(frames)
# y_merge = x_merge_pd.to_numpy()