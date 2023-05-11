import json
import numpy as np
import pandas as pd
from mrc_insar_common.data import data_reader

from dataset import readRas
from config import get_config


def display_all(data, config):
    img_shape = []
    
    for i in range(len(data)):
        rslc1 = data_reader.readBin(data['rslc0'][i], width=config['tiles_size'], dataType='floatComplex') # SLC(RSLC) image
        rslc2 = data_reader.readBin(data['rslc1'][i], width=config['tiles_size'], dataType='floatComplex') # SLC(RSLC) image
        rslc1_label = readRas(data['rslc0_label'][i])[0]
        rslc2_label = readRas(data['rslc1_label'][i])[0]

        rslc1_label[rslc1_label==255] = 1
        rslc2_label[rslc2_label==255] = 1
            
        id = data['rslc1'][i].split("/")[-1].split(".")[0].split("\\")[-1]
        # ifg = np.angle(np.log(np.abs(rslc2))*np.conjugate(np.log(np.abs(rslc1))))

        if rslc1.shape not in img_shape:
            img_shape.append(rslc1.shape)

    print(img_shape)


def display_patch(data):
    img_shape = []
        
    for i in range(len(data)):
        rslc1 = data_reader.readBin(data['rslc0'][i], width=config['tiles_size'], dataType='floatComplex') # SLC(RSLC) image
        rslc2 = data_reader.readBin(data['rslc1'][i], width=config['tiles_size'], dataType='floatComplex') # SLC(RSLC) image
        rslc1_label = readRas(data['rslc0_label'][i])[0]
        rslc2_label = readRas(data['rslc1_label'][i])[0]
    
        rslc1_label[rslc1_label==255] = 1
        rslc2_label[rslc2_label==255] = 1
            
        rslc1_amp = rslc1[data['patch_idx'][i][0]:data['patch_idx'][i][1], data['patch_idx'][i][2]:data['patch_idx'][i][3]]
        rslc2_amp = rslc2[data['patch_idx'][i][0]:data['patch_idx'][i][1], data['patch_idx'][i][2]:data['patch_idx'][i][3]]
        rslc1_label_amp = rslc1_label[data['patch_idx'][i][0]:data['patch_idx'][i][1], data['patch_idx'][i][2]:data['patch_idx'][i][3]]
        rslc2_label_amp = rslc2_label[data['patch_idx'][i][0]:data['patch_idx'][i][1], data['patch_idx'][i][2]:data['patch_idx'][i][3]]
            
        id = data['rslc1'][i].split("/")[-1].split(".")[0].split("\\")[-1]
        # ifg = np.angle(np.log(np.abs(rslc2_amp))*np.conjugate(np.log(np.abs(rslc1_amp))))

        if rslc1_amp.shape not in img_shape:
            img_shape.append(rslc1_amp.shape)


    # print(len(img_shape))
    print(img_shape)






config = get_config()

# train_dir = pd.read_csv(config['train_dir'])
# print("Train examples: ", len(train_dir))
# display_all(train_dir, config)
    
# valid_dir = pd.read_csv(config['valid_dir'])
# print("Valid examples: ", len(valid_dir))
# display_all(valid_dir, config)

# test_dir = pd.read_csv(config['test_dir'])
# print("Test examples: ", len(test_dir))
# display_all(test_dir, config)


# with open("/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/data/json/train_patch_phr_cb_512.json", 'r') as j:
#     patch_dir = json.loads(j.read())
# train_patch_df = pd.DataFrame.from_dict(patch_dir)
# print("Train examples: ", len(train_patch_df))
# display_patch(train_patch_df, 512)

with open("/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/data/json/train_patch_phr_cb_512.json", 'r') as j:
    patch_dir = json.loads(j.read())
train_patch_df = pd.DataFrame.from_dict(patch_dir)
print("Train examples: ", len(train_patch_df))
# display_patch(train_patch_df)

# check the json file index 
patch_index = []
for i in train_patch_df['patch_idx']:
    hig = i[1] - i[0]
    wid = i[3] - i[2]
    if hig != 512 or wid != 512:
        print(i)

    
# with open("/home/mdsamiul/github_project/waterbody_segmentation_complex_data/data/json/valid_patch_phr_512.json", 'r') as j:
#     patch_dir = json.loads(j.read())
# valid_patch_df = pd.DataFrame.from_dict(patch_dir)
# print("Valid examples: ", len(valid_patch_df))
# display_patch(valid_patch_df, config)
    
# with open("/home/mdsamiul/github_project/waterbody_segmentation_complex_data/data/json/test_patch_phr_512.json", 'r') as j:
#     patch_dir = json.loads(j.read())
# test_patch_df = pd.DataFrame.from_dict(patch_dir)
# print("Test examples: ",len(test_patch_df))
# display_patch(test_patch_df, config)