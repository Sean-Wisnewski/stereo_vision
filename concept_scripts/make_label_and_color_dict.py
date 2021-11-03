"""
This file is run once to create a dict of {object# : label} for later use
"""
import pickle
import numpy as np
fname = "../labels/coco2017/labels.txt"
save_name = "../labels/coco2017/labels_dict.pkl"
colors_save_name = "../labels/coco2017/colors_arr.pkl"
count = 1
label_dict = {}
colors_lst = []
with open(fname, 'r') as f:
    lines = f.readlines()
    for line in lines:
        label_dict[count] = line.strip()
        colors_lst.append(np.random.randint(0,256, 3))
        count += 1
colors_arr =np.array(colors_lst)
# use protocol=3 for compatibility with python 3.7 (installed on the nano)
with open(save_name, 'wb') as pkl_file:
    pickle.dump(label_dict, pkl_file, protocol=3)
with open(colors_save_name, 'wb') as pkl_file:
    pickle.dump(colors_arr, pkl_file, protocol=3)
