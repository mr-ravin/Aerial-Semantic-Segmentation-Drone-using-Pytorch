import glob
import random
import os
import numpy as np

grouped_class_id = {
                    0: [0, 6, 10, 11, 12, 13, 14, 21, 22, 23], # --> 0: [155,38,182], 'obstacles'
                    1: [5, 7], # --> 1: [14,135,204], 'water'
                    2: [2, 3, 8, 19, 20], # --> 2: [124,252,0], 'nature'
                    3: [15, 16, 17, 18], # --> 3: [255,20,147], 'moving'
                    4: [1, 4, 9] # --> 4: [169,169,169], 'landable'
                    }

class_id_to_group_id = {e:k for k,v in grouped_class_id.items() for e in v }

def convert_class_id_to_group_id(class_id):
    return class_id_to_group_id[class_id]


def generate_group_mask(raw_mask_np, individual_class_count=24):
    raw_mask_np_shape = raw_mask_np.shape
    group_mask = np.zeros((raw_mask_np_shape[0], raw_mask_np_shape[1]),dtype=np.uint8)
    for idx in range(individual_class_count):
        tmp_mask = (raw_mask_np == idx)*1
        tmp_mask = tmp_mask*class_id_to_group_id[idx]
        group_mask = group_mask + tmp_mask.copy()
    group_mask = np.uint8(group_mask)
    return group_mask


def check_data_consistency(root_dir):
    bgr_filename_list = glob.glob(root_dir+"/dataset/images/*jpg")
    mask_filename_list = glob.glob(root_dir+"/dataset/masks/*png")
    for bgr_filename in bgr_filename_list:
        tmp_name = bgr_filename.split("/")[-1]
        mask_name = tmp_name.replace(".jpg",".png")
        mask_path = root_dir+"/dataset/masks/"+mask_name
        if mask_path not in mask_filename_list : # for any bgr image when its mask does not exist.
            print("removing: ", bgr_filename, assumed_mask_path)
            os.system("rm "+bgr_filename) # remove the bgr image
        

def data_split(root_dir, split_ratio={"train":0.85,"val":0.15}):
    check_data_consistency(root_dir)
    filename_list = glob.glob(root_dir+"/dataset/images/*jpg")
    random.shuffle(filename_list)
    len_filename_list = len(filename_list)
    processed = 0
    for mode in ["train", "val"]:
        if os.path.exists(root_dir+"/dataset/"+mode):
            if os.path.exists(root_dir+"/dataset/"+mode+"/images"): 
                print("Cleaning Previous Splits")
                os.system("rm  "+root_dir+"/dataset/"+mode+"/images/*jpg")
            else:
                os.system("mkdir "+root_dir+"/dataset/"+mode+"/images")
        else:
            os.system("mkdir "+root_dir+"/dataset/"+mode)
        print("Total samples in dataset: ", len(filename_list))
        for rgb_filename in filename_list[processed:processed+int(len_filename_list * split_ratio[mode])]:
            os.system("cp "+rgb_filename+" "+root_dir+"/dataset/"+mode+"/images/")
        processed += int(len_filename_list * split_ratio[mode])
    print("Images in train set: ",len(glob.glob(root_dir+"/dataset/train/images/*jpg")))
    print("Images in val set: ",len(glob.glob(root_dir+"/dataset/val/images/*jpg")))