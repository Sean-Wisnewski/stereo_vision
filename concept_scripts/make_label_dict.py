"""
This file is run once to create a dict of {object# : label} for later use
"""
import pickle
fname = "../labels/coco2017/labels.txt"
save_name = "../labels/coco2017/labels_dict.pkl"
count = 1
label_dict = {}
with open(fname, 'r') as f:
    lines = f.readlines()
    for line in lines:
        label_dict[count] = line
        count += 1
with open(save_name, 'wb') as pkl_file:
    pickle.dump(label_dict, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
