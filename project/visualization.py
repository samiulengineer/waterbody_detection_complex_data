import os
import glob
import yaml
import pathlib
import numpy as np
from utils import get_config_yaml
import pandas as pd
from tiles import readRas
from mrc_insar_common.data import data_reader
# from dataset import get_test_dataloader, read_img, transform_data
import earthpy.plot as ep
import earthpy.spatial as es
from matplotlib import pyplot as plt


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

def total_pixel(data):
    masks = data["masks"]
    pixels = {"Water":0, "NON-Water":0}
    for i in range(len(masks)):
        mask = read_img(masks[i], label=True)
        pixels["Water"] += np.sum(mask)
        pixels["NON-Water"] += (mask.shape[0]*mask.shape[1]) - np.sum(mask)
    return pixels


if __name__=='__main__':
    
    # config = get_config_yaml('config.yaml', {})
    config = get_config_yaml('config.yaml', {})
    print("Saving figures in {}".format(config["visualization_dir"]))
    display_raw_data(config)

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

