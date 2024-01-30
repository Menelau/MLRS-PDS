import glob
from numpy import mean
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import pickle
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def random_forest(x_train, y_train, model_name):
    random_forest_model = RandomForestClassifier(random_state=42)

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = random_forest_model.fit(x_train, y_train)
    pkl_filename = "models/{}/split_{}/random_forest.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


    y_pred = random_forest_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy

# bagging_with_decision_tree
def bagging_with_decision_tree(x_train, y_train, model_name):
    dt = DecisionTreeClassifier(max_depth=None)
    Bagging_classifiers_DecisionTree_AsBaseEstimator = BaggingClassifier(base_estimator=dt,n_estimators=100)

    # print(model_name.split('_')[0])
    # print(model_name.split('_')[1])
    # print(model_name.split('_')[2])
    # print(model_name.split('_')[3].split('.')[0])
    # print(model_name)
    # print(model_name.split('\\')[1])
    # print("processed_"+ model_name.split(".")[0].split("\\")[1])


    split_num = model_name.split('_')[3].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Bagging_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    pkl_filename = "models/{}/split_{}/bagging_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = Bagging_classifiers_DecisionTree_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-decision tree base classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def bagging_with_perceptron(x_train, y_train, model_name):
    perc = Perceptron(shuffle=True)
    model = BaggingClassifier(base_estimator=perc,n_estimators=100)  # Instantiate a BaggingClassifier 'Bagging_classifiers_Perceptron_AsBaseEstimator'

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model.fit(x_train, y_train)
    pkl_filename = "models/{}/split_{}/bagging_with_perceptron.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    # knorae = KNORAE(model).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_pred = knorae.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('{:.4f}'.format(accuracy)) # rondesh karde
    return accuracy


def boosting_with_decision_tree(x_train, y_train, model_name):
    bdt = DecisionTreeClassifier(random_state=42, max_depth=1)
    boosting_classifiers_DecisionTree_AsBaseEstimator = AdaBoostClassifier(base_estimator=bdt, n_estimators=100,
                                                                           random_state=0,
                                                                           algorithm='SAMME')

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = boosting_classifiers_DecisionTree_AsBaseEstimator.fit(x_train, y_train)
    pkl_filename = "models/{}/split_{}/boosting_with_decision_tree.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = boosting_classifiers_DecisionTree_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def Boosting_with_perceptron(x_train, y_train, model_name):
    bperc = Perceptron(tol=1e-3, random_state=42)
    Boosting_classifiers_perceptron_AsBaseEstimator = AdaBoostClassifier(base_estimator=bperc, n_estimators=100,
                                                                         random_state=0, algorithm='SAMME')

    split_num = model_name.split('_')[4].split('.')[0]
    portion = model_name.split('\\')[1]
    model = Boosting_classifiers_perceptron_AsBaseEstimator.fit(x_train, y_train)
    pkl_filename = "models/{}/split_{}/boosting_with_perceptron.pkl".format(portion, split_num)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = Boosting_classifiers_perceptron_AsBaseEstimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy of Bagging-perceptron base Classifier: {:.3f}'.format(accuracy)) # rondesh karde
    return accuracy


def save_obj(obj, name):
    with open('{}.pkl'.format(name), 'wb') as configFile:
        pickle.dump(obj, configFile)


dict_results = {}
dict_sd_pc = {}
for files in glob.glob('proccc/*'):
    print(files)
    pc = []
    sd_pc = []
    for items in glob.glob("{}/*.*".format(files)):
        print(items)
        # print(len(items))

        data = np.load(items, allow_pickle=True)
        # print(data)
        input_list = data.tolist()  # How to get x_train
        x_train = input_list['FrameStack'][0]
        x_test = input_list['FrameStack'][1]
        y_train = input_list['FrameStack'][2]
        y_test = input_list['FrameStack'][3]
        x_dsel = input_list['FrameStack'][4]
        y_dsel = input_list['FrameStack'][5]
        print(len(x_train))
        print(len(x_test))
        print(len(x_dsel))
        print(len(y_train))
        print(len(y_test))
        print(len(y_dsel))

        defff = bagging_with_decision_tree
        # defff = bagging_with_perceptron
        # defff = boosting_with_decision_tree
        # defff = Boosting_with_perceptron
        # defff = random_forest

        accur = defff(x_train, y_train, items)
        pc.append(accur)
        dict_results["{}".format(files)] = '{:.4f}'.format(mean(pc))
        np.std(pc, dtype=np.float64)
        sd_pc.append(np.std(pc, dtype=np.float64))

    dict_sd_pc["{}".format(files)] = '{:.4f}'.format(mean(sd_pc))
# print(dict_results)
print(dict_results.values())
print(dict_sd_pc.values())

#save model
save_obj(dict_results, "models")





