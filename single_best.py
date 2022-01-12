import glob
from numpy import mean
import pickle
import numpy as np
from deslib.static.single_best import SingleBest

dict_sb_results = {}
dict_standard_deviation = {}

for files in glob.glob('processed_8/*'):
    print(files)

    single_best_results = []
    sd_single_best = []
    for items in glob.glob("{}/*.*".format(files)):
        # print(items)
        file_name = files.split("\\")[1]
        # print(file_name)
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        print(item_name)
        ss = file_name.split("_")[0]
        model_load = glob.glob("models_f/{}/split_{}/bagging_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/bagging_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/boosting_with_decision_tree.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/boosting_with_perceptron.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("models_f/{}/split_{}/random_forest.pkl".format(file_name, item_name))[0]
        # model_load = glob.glob("Models_FLT_f/{}/{}_{}.pkl".format(file_name, ss, item_name))[0]
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

            pool_classifiers = my_model

            sb_bdt = SingleBest(pool_classifiers = pool_classifiers, scoring = None, random_state = None, n_jobs = -1)
            sb_bdt.fit(x_dsel, y_dsel)
            sb_bdt.predict(x_dsel)
            acc = sb_bdt.score(x_test, y_test, sample_weight=None)
            single_best_results.append(acc)

    dict_sb_results["{}".format(files)] = '{:.4f}'.format(mean(single_best_results))
    print(dict_sb_results)
    np.std(single_best_results, dtype=np.float64)
    sd_single_best.append(np.std(single_best_results, dtype=np.float64))
    dict_standard_deviation["{}".format(files)] = '{:.4f}'.format(mean(sd_single_best))
    print(dict_standard_deviation)

print(dict_sb_results.values())
print(dict_standard_deviation.values())