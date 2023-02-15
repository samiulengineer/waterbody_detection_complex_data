import os
import glob
import yaml
import cv2
import json
import pathlib
import numpy as np
import pandas as pd
import earthpy.plot as ep
import earthpy.spatial as es
from matplotlib import pyplot as plt

from tiles import readRas
from mrc_insar_common.data import data_reader
from utils import get_config_yaml
from dataset import read_img


# setup gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def class_balance_check(patchify, data_dir):
    """
    Summary:
        checking class percentage in full dataset
    Arguments:
        patchify (bool): TRUE if want to check class balance for patchify experiments
        data_dir (str): directory where data files save
    Return:
        class percentage
    """
    if patchify:
        with open(data_dir, 'r') as j:
            train_data = json.loads(j.read())
        labels = train_data['masks']
        patch_idx = train_data['patch_idx']
    else:
        train_data = pd.read_csv(data_dir)
        labels = train_data.masks.values
        patch_idx = None
    
    total = 0
    class_name = {}

    for i in range(len(labels)):
        mask = cv2.imread(labels[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if patchify:
            idx = patch_idx[i]
            mask = mask[idx[0]:idx[1], idx[2]:idx[3]]
            
        total_pix = mask.shape[0]*mask.shape[1]
        total += total_pix
        
        dic = {}
        keys = np.unique(mask)
        for i in keys:
            dic[i] = np.count_nonzero(mask == i)
            
        for key, value in dic.items(): 
            if key in class_name.keys():
                class_name[key] = value + class_name[key]
            else:
                class_name[key] = value
        
    for key, val in class_name.items():
        class_name[key] = (val/total)*100
        
    print("Class percentage:")
    for key, val in class_name.items():
        print("class pixel: {} = {}".format(key, val))
        


def check_height_width(data_dir):
    """
    Summary:
        check unique hight and width of images from dataset
    Arguments:
        data_dir (str): path to csv file
    Return:
        print all the unique height and width
    """
    
    data = pd.read_csv(data_dir)
    # removing UU or UMM or UM
    # data = data[data['feature_ids'].str.contains('uu_00') == False]
    data = data[data['feature_ids'].str.contains('umm_00') == False]
    data = data[data['feature_ids'].str.contains('um_00') == False]
    
    
    print("Dataset:  ", data.shape)
    
    input_img = data.feature_ids.values
    input_mask = data.masks.values
    
    input_img_shape = []
    input_mask_shape = []
    
    for i in range(len(input_img)):
        img = cv2.imread(input_img[i])
        mask = cv2.imread(input_mask[i])

        if img.shape not in input_img_shape:
            input_img_shape.append(img.shape)

        if mask.shape not in input_mask_shape:
            input_mask_shape.append(mask.shape)

    print("Input image shapes: ", input_img_shape)
    print("Input mask shapes: ", input_mask_shape)

        
def plot_curve(models, metrics, fname):
    """
    Summary:
        plot curve between metrics and model
    Arguments:
        models (list): list of model names
        metrics (dict): dictionary containing the metrics name and conrresponding value
        fname (str): name of the figure
    Return:
        figure
    """
    keys = list(metrics.keys())
    val = list(metrics.values())
    threshold = np.arange(0,len(models),1)
    colorstring = 'bgrcmykw'
    # markerstring = [ '-', '.', 'o', '*', 'x']

    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.title("Experimental result for different models", fontsize=30, fontweight="bold")

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontweight('bold')
    
    for i in range(len(keys)):
        plt.plot(threshold, val[i], color=colorstring[i], linewidth=3.0, marker='o', markersize=10, label=keys[i])

    # plt.legend(loc='best')
    ax.legend(prop=dict(weight='bold', size=18), loc='best')
    
    plt.xlabel("Models", fontweight="bold")
    plt.ylabel("Metrics score", fontweight="bold")
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontweight('bold')
    plt.xticks(ticks=threshold,labels=models)
    plt.savefig(fname, bbox_inches='tight',  dpi = 1000)
    plt.show()


def plot(display_list, id, directory):
    
    plt.figure(figsize=(12, 8))
    title = list(display_list.keys())

    for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            
            # plot dem channel using earthpy
            if title[i]=="dem":
                ax = plt.gca()
                hillshade = es.hillshade(display_list[title[i]], azimuth=180)
                ep.plot_bands(
                    display_list[title[i]],
                    cbar=False,
                    cmap="terrain",
                    title=title[i],
                    ax=ax
                )
                ax.imshow(hillshade, cmap="Greys", alpha=0.5)
            
            # gray image plot rslc
            elif title[i]=="rslc1" or title[i]=="rslc2":
                plt.title(title[i])
                c2 = plt.imshow((display_list[title[i]])**0.3, cmap="gray", interpolation=None)
                ax = plt.gca()
                fig = plt.gcf()
                fig.colorbar(c2, ax=ax, fraction=0.046)
                plt.axis('off')
            
            # ifr plot
            elif title[i]=="ifg":
                plt.title(title[i])
                c2 = plt.imshow((display_list[title[i]]), cmap="jet", interpolation=None)
                ax = plt.gca()
                fig = plt.gcf()
                fig.colorbar(c2, ax=ax, fraction=0.046)
                plt.axis('off')

            # plot labels
            else:
                plt.title(title[i])
                c2 = plt.imshow((display_list[title[i]]), vmin=0, vmax=1)
                ax = plt.gca()
                fig = plt.gcf()
                fig.colorbar(c2, ax=ax, fraction=0.046)
                plt.axis('off')

    prediction_name = "img_id_{}.png".format(id) # create file name to save
    plt.savefig(os.path.join(directory, prediction_name), bbox_inches='tight', dpi=800)
    plt.clf()
    plt.cla()
    plt.close()


def display_all(data, config):
    """
    Summary:
        save all images into single figure
    Arguments:
        data : data file holding images path
        directory (str) : path to save images
    Return:
        save images figure into directory
    """
    
    pathlib.Path((config['visualization_dir']+'display')).mkdir(parents = True, exist_ok = True)

    for i in range(len(data)):
        rslc1 = data_reader.readBin(data['rslc0'][i], width=512, dataType='floatComplex') # SLC(RSLC) image
        rslc2 = data_reader.readBin(data['rslc1'][i], width=512, dataType='floatComplex') # SLC(RSLC) image
        rslc1_label = readRas(data['rslc0_label'][i])[0]
        rslc2_label = readRas(data['rslc1_label'][i])[0]

        rslc1_label[rslc1_label==255] = 1
        rslc2_label[rslc2_label==255] = 1
        
        id = data['rslc1'][i].split("/")[-1].split(".")[0].split("\\")[-1]
        
        display_list = {
                     "rslc1":np.abs(rslc1),
                     "rslc2":np.abs(rslc2),
                     "ifg":np.angle(rslc1*np.conjugate(rslc2)),
                     "rslc1_label": rslc1_label,
                     "rslc2_label":rslc2_label
                     }
        plot(display_list, id, (config['visualization_dir']+'display'))


        

def display_raw_data(config):

    pathlib.Path((config['visualization_dir']+'display_raw_data')).mkdir(parents = True, exist_ok = True)

    wid, hgt = 11361, 10820  # from the masterpar.xml

    rslcs = sorted(glob.glob(config["dataset_dir"]+"rslc/*.rslc.notopo"))
    label = sorted(glob.glob(config["dataset_dir"]+"label/*.ras"))

    rslc1 = data_reader.readBin(rslcs[0], width=wid, dataType='floatComplex') # SLC(RSLC) image
    rslc2 = data_reader.readBin(rslcs[1], width=wid, dataType='floatComplex') # SLC(RSLC) image
    rslc1_label = readRas(label[0])[0]
    rslc2_label = readRas(label[1])[0]

    rslc1_label[rslc1_label==255] = 1
    rslc2_label[rslc2_label==255] = 1

    id = rslcs[0].split("/")[-1].split(".")[0].split("\\")[-1]
    id2 = rslcs[3].split("/")[-1].split(".")[0].split("\\")[-1]

    display_list = {
                     "rslc1": np.abs(rslc1),
                     "rslc2": np.abs(rslc2),
                     "ifg": np.angle(rslc2*np.conjugate(rslc1)),
                     "rslc1_label": rslc1_label,
                     "rslc2_label": rslc2_label
                     }
    plot(display_list, (id+"_"+id2), (config['visualization_dir']+'display_raw_data'))

def total_pixel(config):
    label = sorted(glob.glob(config["dataset_dir"]+"label/*.ras"))
    rslc1_label = readRas(label[0])[0]

    rslc1_label[rslc1_label==255] = 1
    total_pix = rslc1_label.shape[0]*rslc1_label.shape[1]
    pixels = {"Water":np.sum(rslc1_label)/total_pix, 
              "NON-Water":((rslc1_label.shape[0]*rslc1_label.shape[1]) - np.sum(rslc1_label))/total_pix}
    return pixels
    

if __name__ == '__main__':

    config = get_config_yaml('project/config.yaml', {})

    pathlib.Path(config['visualization_dir']).mkdir(
        parents=True, exist_ok=True)
    
    # # check for label in the dataset. For patchify pass True with json file. For real image pass False with the csv file
    # class_balance_check(False, "/home/mdsamiul/github_project/road_segmentation/data/train.csv")
    # class_balance_check(True, "/home/mdsamiul/github_project/road_segmentation/data/json/train_patch_phr_cb_256.json")
    
    
    # # check shape of input image and mask
    # check_height_width("/home/mdsamiul/github_project/road_segmentation/data/train.csv")
    
    
    # metrics result plot
    # models = ['Fapnet', 'UNet', 'UNet++', 'VNet', 'U2Net', 'DNCNN', 'FPN', 'LINKNET', 'ATTUNET', 'R2UNET']
    # metrics = {
    #     'MIOU':      [0.96, 0.76, 0.78, 0.77, 0.61, 0.76, 0.83, 0.84, 0.86, 0.86],
    #     'F-1 score': [0.98, 0.91, 0.91, 0.87, 0.93, 0.91, 0.95, 0.96, 0.96, 0.94],
    #     'Precision': [0.98, 0.70, 0.87, 0.66, 0.50, 0.68, 0.87, 0.84, 0.88, 0.77],
    #     'Recall':    [0.98, 0.74, 0.63, 0.71, 0.80, 0.74, 0.90, 0.95, 0.95, 0.89]
    # }
    # fname = config['visualization_dir'] + 'metrics_result.jpg'
    # plot_curve(models, metrics, fname)
    
    
    # print(total_pixel(config))
    
    # print("Saving figures in {}".format(config["visualization_dir"]))
    # display_raw_data(config)

    # train_dir = pd.read_csv(config['train_dir'])
    # print("Train examples: ", len(train_dir))

    # test_dir = pd.read_csv(config['test_dir'])
    # print("Test examples: ", len(test_dir))

    # valid_dir = pd.read_csv(config['valid_dir'])
    # print("Valid examples: ", len(valid_dir))
    
    # print("Saving figures....")
    # display_all(train_dir, config)
    # display_all(valid_dir, config)
    # display_all(test_dir, config)
