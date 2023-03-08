import pandas as pd
import os
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from pymfe.mfe import MFE
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

with open('predictions_phase2_BDT_2NN.pkl', 'rb') as f:
    predictions = pickle.load(f)

DS = [prediction.split("_")[1] for prediction in predictions]
print(DS)

with open('predictions_phase1_OLA_2NN.pkl', 'rb') as f:
    OLA1 = pickle.load(f)
with open('predictions_phase1_DESMI_2NN.pkl', 'rb') as f:
    DESMI1 = pickle.load(f)
with open('predictions_phase1_DESP_2NN.pkl', 'rb') as f:
    DESP1 = pickle.load(f)
with open('predictions_phase1_KNORAE_2NN.pkl', 'rb') as f:
    KNORAE1 = pickle.load(f)
with open('predictions_phase1_KNORAU_2NN.pkl', 'rb') as f:
    KNORAU1 = pickle.load(f)
    # print(KNORAU1)
with open('predictions_phase1_METADES_2NN.pkl', 'rb') as f:
    METADES1 = pickle.load(f)
with open('predictions_phase1_MLA_2NN.pkl', 'rb') as f:
    MLA1 = pickle.load(f)

# print(pool)
OLA = [OLA1.split("_")[0] for OLA1 in OLA1]
# print(OLA)
DESMI = [DESMI1.split("_")[0] for DESMI1 in DESMI1]
# print(DESMI)
DESP = [DESP1.split("_")[0] for DESP1 in DESP1]
# print(DESP)
KNORAE = [KNORAE1.split("_")[0] for KNORAE1 in KNORAE1]
# print(KNORAE)
KNORAU = [KNORAU1.split("_")[0] for KNORAU1 in KNORAU1]
print(KNORAU)
METADES = [METADES1.split("_")[0] for METADES1 in METADES1]
# print(RF)
MLA = [MLA1.split("_")[0] for MLA1 in MLA1]
# print(FLT)

pool_DS = []
for i in range(len(DS)):
    algorithm_name = DS[i]

    df_algorithm = []
    if algorithm_name == 'OLA':
        df_algorithm = OLA
    elif algorithm_name == 'DESMI':
        df_algorithm = DESMI
    elif algorithm_name == 'DESP':
        df_algorithm = DESP
    elif algorithm_name == 'KNORAE':
        df_algorithm = KNORAE
    elif algorithm_name == 'KNORAU':
        df_algorithm = KNORAU
    elif algorithm_name == 'METADES':
        df_algorithm = METADES
    elif algorithm_name == 'MLA':
        df_algorithm = MLA

    # inja corresponding element ezafe mishe be list
    pool_DS.append(df_algorithm[i] + '_' + algorithm_name)

print(pool_DS)


df3 = pd.read_excel('dataframe_friedman.xlsx')
df4 = df3.iloc[:, 1:]
print(df4)
majority_votings = df4.iloc[:, [0, 8, 16, 24, 32, 40, 48]]  # columns to be removed
df5 = df4.drop(columns=majority_votings)
print(df5)

algorithm = df5
highest_value_col_namee = algorithm.idxmax(axis=1)
print(highest_value_col_namee)
original_listt = highest_value_col_namee

accuracy = accuracy_score(original_listt , pool_DS)
print(accuracy)   #24.65
