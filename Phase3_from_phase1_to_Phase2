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

with open('predictions_phase1_KNORAE_2NN.pkl', 'rb') as f:
    predictions = pickle.load(f)

pool = [prediction.split("_")[0] for prediction in predictions]
print(pool)

with open('predictions_phase2_LIT_2NN.pkl', 'rb') as f:
    LIT1 = pickle.load(f)
with open('predictions_phase2_BP_2NN.pkl', 'rb') as f:
    BP1 = pickle.load(f)
with open('predictions_phase2_BDT_2NN.pkl', 'rb') as f:
    BDT1 = pickle.load(f)
with open('predictions_phase2_BSDT_2NN.pkl', 'rb') as f:
    BSDT1 = pickle.load(f)
with open('predictions_phase2_BSP_2NN.pkl', 'rb') as f:
    BSP1 = pickle.load(f)
with open('predictions_phase2_RF_2NN.pkl', 'rb') as f:
    RF1 = pickle.load(f)
with open('predictions_phase2_FLT_2NN.pkl', 'rb') as f:
    FLT1 = pickle.load(f)

# print(pool)
LIT = [LIT1.split("_")[1] for LIT1 in LIT1]
# print(LIT)
BP = [BP1.split("_")[1] for BP1 in BP1]
# print(BP)
BDT = [BDT1.split("_")[1] for BDT1 in BDT1]
# print(BDT)
BSDT = [BSDT1.split("_")[1] for BSDT1 in BSDT1]
# print(BSDT)
BSP = [BSP1.split("_")[1] for BSP1 in BSP1]
# print(BSP)
RF = [RF1.split("_")[1] for RF1 in RF1]
# print(RF)
FLT = [FLT1.split("_")[1] for FLT1 in FLT1]
# print(FLT)


pool_DS = []
for i in range(len(pool)):
    algorithm_name = pool[i]

    if algorithm_name == 'LIT':
        df_algorithm = LIT
    elif algorithm_name == 'BP':
        df_algorithm = BP
    elif algorithm_name == 'BDT':
        df_algorithm = BDT
    elif algorithm_name == 'BSDT':
        df_algorithm = BSDT
    elif algorithm_name == 'BSP':
        df_algorithm = BSP
    elif algorithm_name == 'RF':
        df_algorithm = RF
    elif algorithm_name == 'FLT':
        df_algorithm = FLT

    # Hesam corresponding element be list ezafe mishe inja
    pool_DS.append(algorithm_name + '_' + df_algorithm[i])

print(pool_DS)


df3 = pd.read_excel('dataframe_friedman.xlsx')

# esme data ha ro hazf kon vase rank dadan
df4 = df3.iloc[:, 1:]
# df2 = df1.drop(index=0)
print(df4)
majority_votings = df4.iloc[:, [0, 8, 16, 24, 32, 40, 48]]  # columns to be removed
df5 = df4.drop(columns=majority_votings)
print(df5)

algorithm = df5
highest_value_col_namee = algorithm.idxmax(axis=1)
print(highest_value_col_namee)
original_listt = highest_value_col_namee

accuracy = accuracy_score(original_listt , pool_DS)
print(accuracy)   #22.91
