import pickle
import os
from report import Report
import pandas as pd


def read_pickle(path):
    for file_name in os.scandir(path):
        with open(file_name) as f:
            return pickle.load(f)


if __name__ == "__main__":
    results_table = {}
    path = "/Users/Hesam/main_repository/Hesam/MP4_EOL/report"


    for file_name in os.listdir(path):
        if file_name.endswith(".rep"):

            eole_results = Report.load(path+"/"+file_name)
            eole_accuracy = eole_results.accuracy_sample[:, -1]
            results_table[file_name] = (eole_accuracy.mean(), eole_accuracy.std())
    results_df = pd.DataFrame.from_dict(results_table, orient='index', columns=["Accuracy Mean", "Accuracy Std"])
    results_df.to_csv(path+"/results.csv",index=True)

