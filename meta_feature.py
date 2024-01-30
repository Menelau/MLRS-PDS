from pymfe.mfe import MFE
import glob
import numpy as np
from numpy import mean
import pickle

for files in glob.glob('alll_removed/*'):
    for items in glob.glob("{}/*.*".format(files)):
        # print(files)
        # print(items)


        file_name = files.split("\\")[1]
        # print(file_name)
        item_name = items.split("\\")[2].split(".")[0].split("_")[2]
        print(item_name)
        ss = file_name.split("_")[0]
        # print(ss)

        # rr = ss.split("/")[1] + item_name.split("_")[6]
        # print(rr)
        # hh = item_name.split("_")[1]
        # print(hh)

        data = np.load(items, allow_pickle=True)
        # print(data)
        input_list = data.tolist()  # How to get x_train
        x_train = input_list['FrameStack'][0]
        x_test = input_list['FrameStack'][1]
        y_train = input_list['FrameStack'][2]
        y_test = input_list['FrameStack'][3]
        x_dsel = input_list['FrameStack'][4]
        y_dsel = input_list['FrameStack'][5]

        # Extract all available measures
        mfe = MFE(groups="all")
        # mfe = MFE(groups=['general', 'statistical', 'info-theory', 'Relative Landmarking', 'Subsampling Landmarking', 'Clustering', 'Concept', 'Itemset', 'Complexity', 'all'])
        mfe.fit(x_train, y_train)
        meta_features = mfe.extract()
        print(meta_features)

        # with open("{}_meta_features.pickle".format(items), "wb") as file:
        #     pickle.dump(meta_features, file)
        pkl_filename = "mf/{}/split_{}/meta_features.pkl".format(file_name, item_name)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(meta_features, file)
