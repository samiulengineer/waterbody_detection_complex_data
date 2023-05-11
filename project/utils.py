import os
import json
import math
import glob
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import earthpy.plot as ep
from tensorflow import keras
import earthpy.spatial as es
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import moviepy.video.io.ImageSequenceClip

from loss import *
from dataset import read_img, transform_data


# Callbacks and Prediction during Training
# ----------------------------------------------------------------------------------------------
class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self, val_dataset, model, config):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """
        super(keras.callbacks.Callback, self).__init__()

        self.val_dataset = val_dataset
        self.model = model
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """
        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * \
            math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr

    def on_epoch_end(self, epoch, logs={}):
        """
        Summary:
            call after every epoch to predict mask
        Arguments:
            epoch (int): current epoch
        Output:
            save predict mask
        """
        if (epoch % self.config['val_plot_epoch'] == 0):  # every after certain epochs the model will predict mask
            # save image/images with their mask, pred_mask and accuracy
            if self.config['patchify']:
                val_show_predictions(self.val_dataset, self.model, self.config)
            else:
                show_predictions(self.val_dataset, self.model, self.config, val=True)

    def get_callbacks(self, val_dataset, model):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """
        if self.config['csv']:  # save all type of accuracy in a csv file for each epoch
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(
                self.config['csv_log_dir'], self.config['csv_log_name']), separator=",", append=False))

        if self.config['checkpoint']:  # save the best model
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(
                self.config['checkpoint_dir'], self.config['checkpoint_name']), save_best_only=True))

        if self.config['tensorboard']:  # Enable visualizations for TensorBoard
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(
                self.config['tensorboard_log_dir'], self.config['tensorboard_log_name'])))

        if self.config['lr']:  # adding learning rate scheduler
            self.callbacks.append(
                keras.callbacks.LearningRateScheduler(schedule=self.lr_scheduler))

        if self.config['early_stop']:  # early stop the training if there is no change in loss
            self.callbacks.append(keras.callbacks.EarlyStopping(
                monitor='my_mean_iou', patience=self.config['patience']))

        if self.config['val_pred_plot']:  # plot validated image for each epoch
            self.callbacks.append(SelectCallbacks(
                val_dataset, model, self.config))

        return self.callbacks


# Prepare masks
# ----------------------------------------------------------------------------------------------
def create_mask(mask, pred_mask):
    """
    Summary:
        apply argmax on mask and pred_mask class dimension
    Arguments:
        mask (ndarray): image labels/ masks
        pred_mask (ndarray): prediction labels/ masks
    Return:
        return mask and pred_mask after argmax
    """
    mask = np.argmax(mask, axis = 3)
    pred_mask = np.argmax(pred_mask, axis = 3)
    return mask, pred_mask


# Sub-ploting and save
# ----------------------------------------------------------------------------------------------

def display(display_list, idx, directory, score, exp, evaluation=False, visualize=False):
    
    plt.figure(figsize=(12, 8))  # set the figure size
    title = list(display_list.keys())  # get tittle

    # plot all the image in a subplot
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        # plot rslc 
        if title[i]=="RSLC1 AMP" or title[i]=="RSLC2 AMP":
            plt.title(title[i], fontsize=6)
            c2 = plt.imshow((display_list[title[i]])**0.25, cmap="gray", interpolation=None)
            ax = plt.gca()
            fig = plt.gcf()
            cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
            cbar.ax.tick_params(labelsize=6) 
            plt.axis('off')
        # for ploting prediction mask on input image
        elif 'Prediction' in title[i]:       
            plt.title(title[i])
            masked = np.ma.masked_where(display_list[title[i]] == 0, display_list[title[i]])
            plt.imshow(display_list["image"], 'gray', interpolation='none')
            plt.imshow(masked, 'jet', interpolation='none', alpha=0.8)
            plt.axis('off')
        # ifr plot
        elif title[i]=="IFG":
            plt.title(title[i], fontsize=6)
            c2 = plt.imshow((display_list[title[i]]), cmap="YlGnBu", interpolation=None)
            ax = plt.gca()
            fig = plt.gcf()
            cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
            cbar.ax.tick_params(labelsize=6) 
            plt.axis('off')
        # plot labels
        else:
            plt.title(title[i], fontsize=6)
            c2 = plt.imshow((display_list[title[i]])**0.3, cmap="gray", interpolation=None)
            ax = plt.gca()
            fig = plt.gcf()
            cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
            cbar.ax.tick_params(labelsize=6) 
            plt.axis('off')

    # create file name to save
    if evaluation:
        prediction_name = "{}_{}.png".format(exp, idx)
    elif visualize:
        prediction_name = "img_id_{}.png".format(idx)
    else:
        prediction_name = "{}_{}_miou_{:.4f}.png".format(exp, idx, score) 
    
    plt.savefig(os.path.join(directory, prediction_name),
                bbox_inches='tight', dpi=800)  # save all the figures
    plt.clf()
    plt.cla()
    plt.close()



# def display(display_list, idx, directory, score, exp, evaluation=False, visualize=False):
    
#     plt.figure(figsize=(12, 8))  # set the figure size
#     title = list(display_list.keys())  # get tittle

#     # plot all the image in a subplot
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         if title[i] == "DEM":  # for plot nasadem image channel
#             ax = plt.gca()
#             hillshade = es.hillshade(display_list[title[i]], azimuth=180)
#             ep.plot_bands(
#                 display_list[title[i]],
#                 cbar=False,
#                 cmap="terrain",
#                 title=title[i],
#                 ax=ax
#             )
#             ax.imshow(hillshade, cmap="Greys", alpha=0.5)
#         elif title[i] == "VV" or title[i] == "VH":  # for plot VV or VH image channel
#             plt.title(title[i])
#             plt.imshow((display_list[title[i]]))
#             plt.axis('off')
#         elif 'Prediction' in title[i]:       # for ploting prediction mask on input image
#             plt.title(title[i])
#             masked = np.ma.masked_where(display_list[title[i]] == 0, display_list[title[i]])
#             plt.imshow(display_list["image"], 'gray', interpolation='none')
#             plt.imshow(masked, 'jet', interpolation='none', alpha=0.8)
#             plt.axis('off')
#         # gray image plot rslc
#         elif title[i]=="RSLC1 AMP" or title[i]=="RSLC2 AMP":
#             plt.title(title[i], fontsize=6)
#             c2 = plt.imshow((display_list[title[i]])**0.25, cmap="gray", interpolation=None)
#             ax = plt.gca()
#             fig = plt.gcf()
#             cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
#             cbar.ax.tick_params(labelsize=6) 
#             plt.axis('off')
#         elif title[i]=="rslc1_label" or title[i]=="rslc2_label":
#             plt.title(title[i], fontsize=6)
#             c2 = plt.imshow((display_list[title[i]])**0.3, cmap="gray", interpolation=None)
#             ax = plt.gca()
#             fig = plt.gcf()
#             cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
#             cbar.ax.tick_params(labelsize=6) 
#             plt.axis('off')
            
#         # ifr plot
#         elif title[i]=="IFG":
#             plt.title(title[i], fontsize=6)
#             c2 = plt.imshow((display_list[title[i]]), cmap="YlGnBu", interpolation=None)
#             ax = plt.gca()
#             fig = plt.gcf()
#             cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
#             cbar.ax.tick_params(labelsize=6) 
#             plt.axis('off')

#         # plot labels
#         else:
#             plt.title(title[i], fontsize=6)
#             c2 = plt.imshow((display_list[title[i]]), vmin=0, vmax=1)
#             ax = plt.gca()
#             fig = plt.gcf()
#             cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
#             cbar.ax.tick_params(labelsize=6) 
#             plt.axis('off')

#     # create file name to save
#     if evaluation:
#         prediction_name = "{}_{}.png".format(exp, idx)
#     elif visualize:
#         prediction_name = "img_id_{}.png".format(idx)
#     else:
#         prediction_name = "{}_{}_miou_{:.4f}.png".format(exp, idx, score) 
    
#     plt.savefig(os.path.join(directory, prediction_name),
#                 bbox_inches='tight', dpi=800)  # save all the figures
#     plt.clf()
#     plt.cla()
#     plt.close()
    
    
def display_label(img, img_path, directory):
    """
    Summary:
        save only predicted labels
    Arguments:
        img (np.array): predicted label
        img_path (str) : source image path
        directory (str): saving directory
    Return:
        save images figure into directory
    """
    
    img_path_split = os.path.split(img_path)
    
    if 'umm_' in img_path_split[1]:
        img_name = img_path_split[1][ : 4] + 'road_' + img_path_split[1][4 : ]
    elif 'um_' in img_path_split[1]:
        img_name = img_path_split[1][ : 3] + 'lane_' + img_path_split[1][3 : ]
    else:
        img_name = img_path_split[1][ : 3] + 'road_' + img_path_split[1][3 : ]
    
    plt.imsave(directory+'/'+img_name, img)
    

# Combine patch images and save
# ----------------------------------------------------------------------------------------------

# plot single will not work here
def patch_show_predictions(dataset, model, config):
    """
    Summary:
        predict patch images and merge together during test and evaluation
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
    Return:
        merged patch image
    """
    
    # predict patch images and merge together
    if config["evaluation"]:
        var_list = ["eval_dir", "p_eval_dir"]
    else:
        var_list = ["test_dir", "p_test_dir"]

    with open(config[var_list[1]], 'r') as j:  # opening the json file
        patch_dir = json.loads(j.read())

    df = pd.DataFrame.from_dict(patch_dir)  # read as panadas dataframe
    full_img_dir = pd.read_csv(config[var_list[0]])  # get the csv file
    total_score = 0.0

    # loop to traverse full dataset
    for i in range(len(full_img_dir)):
        # get tiles size
        mask_size = config["height"]
        # for same mask directory get the index
        idx = df[df["rslc0_label"] == full_img_dir["rslc0_label"][i]].index

        # construct a single full image from prediction patch images
        pred_full_label1 = np.zeros((mask_size, mask_size), dtype=int)
        pred_full_label2 = np.zeros((mask_size, mask_size), dtype=int)
        for j in idx:
            p_idx = patch_dir["patch_idx"][j]
            feature, _, _, _ = dataset.get_random_data(j)
            pred1, pred2 = model.predict(feature)
            pred1 = np.argmax(pred1, axis=3)
            pred2 = np.argmax(pred2, axis=3)
            pred_full_label1[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred1[0] 
            pred_full_label2[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred2[0]   # [start hig: end index, ]

        # get full feature image and mask
        full_feature , full_mask1, full_mask2 = read_img(full_img_dir.iloc[i], width=config['tiles_size'])
        
        # calculate keras MeanIOU score
        m1 = keras.metrics.MeanIoU(num_classes=config['num_classes'])
        m2 = keras.metrics.MeanIoU(num_classes=config['num_classes'])
        m1.update_state(full_mask1, pred_full_label1)
        m2.update_state(full_mask2, pred_full_label2)
        score1 = m1.result().numpy()
        score2 = m2.result().numpy()
        total_score = (score1 + score2) / 2

        # plot and saving image
        if config["evaluation"]:
            # display({"image": feature_img,      # change in the key "image" will have to change in the display
            #          #"mask": pred_full_label,
            #      "Prediction": pred_full_label
            #      }, i, config['prediction_eval_dir'], score, config['experiment'], config["evaluation"])
            
            # use this function only to save predicted image
            display_label(pred_full_label1, full_img_dir.iloc[i], config['prediction_eval_dir'])
        else:
            display({"RSLC1 AMP": full_feature[:,:,0],
                    "RSLC2 AMP ": full_feature[:,:,1],
                    "IFG": full_feature[:,:,2],
                    "RSLC1 Mask": full_mask1,
                    "RSLC2 Mask": full_mask2,
                    "RSLC1 (MeanIOU_{:.4f})".format(score1): pred_full_label1,
                    "RSLC2 (MeanIOU_{:.4f})".format(score2): pred_full_label2
                    }, i, config['prediction_test_dir'], total_score, config['experiment'])


# Save all plot figures
# ----------------------------------------------------------------------------------------------
def show_predictions(dataset, model, config, val=False):
    """
    Summary: 
        save image/images with their mask, pred_mask and accuracy
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
        val (bool): for validation plot save
    Output:
        save predicted image/images
    """

    if val:
        directory = config['prediction_val_dir']
    else:
        directory = config['prediction_test_dir']

    # save single image after prediction from dataset
    if config['plot_single']:
        data = dataset.get_random_data(config['index'])
        data = [data]
    else:
        data = dataset
        idx = 0
    
    print(len(data))
    for feature, mask1, mask2, idx in data: # save all image prediction in the dataset
        # feature, mask1, mask2, idx = data_p
        pred1, pred2 = model.predict_on_batch(feature)
        mask1, pred_mask1 = create_mask(mask1, pred1)
        mask2, pred_mask2 = create_mask(mask2, pred2)
        for i in range(len(feature)): # save single image prediction in the batch
            m1 = keras.metrics.MeanIoU(num_classes=config['num_classes'])
            m2 = keras.metrics.MeanIoU(num_classes=config['num_classes'])
            m1.update_state(mask1[i], pred_mask1[i])
            m2.update_state(mask2[i], pred_mask2[i])
            score1 = m1.result().numpy()
            score2 = m2.result().numpy()
            display({"RSLC1 AMP": feature[i][:,:,0],
                     "RSLC2 AMP ": feature[i][:,:,1],
                     "IFG": feature[i][:,:,2],
                      "RSLC1 Mask": mask1[i],
                      "RSLC2 Mask": mask2[i],
                      "RSLC1 (MeanIOU_{:.4f})".format(score1): pred_mask1[i],
                      "RSLC2 (MeanIOU_{:.4f})".format(score2): pred_mask2[i]
                      }, idx, directory, 0, config['experiment'])
            idx += 1
            
            
# validation full image plot
# ----------------------------------------------------------------------------------------------
def val_show_predictions(dataset, model, config):
    """
    Summary:
        predict patch images and merge together during training
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
    Return:
        merged patch image
    """
    var_list = ["valid_dir", "p_valid_dir"]

    with open(config[var_list[1]], 'r') as j:  # opening the json file
        patch_dir = json.loads(j.read())

    df = pd.DataFrame.from_dict(patch_dir)  # read as panadas dataframe
    full_img_dir = pd.read_csv(config[var_list[0]])  # get the csv file

    i = random.randint(0, len(full_img_dir))
    # get tiles size
    mask_size = config["tiles_size"]
    # for same mask directory get the index
    idx = df[df["rslc0_label"] == full_img_dir["rslc0_label"][i]].index

    # construct a single full image from prediction patch images
    pred_full_label1 = np.zeros((mask_size, mask_size), dtype=int)
    pred_full_label2 = np.zeros((mask_size, mask_size), dtype=int)
    for j in idx:
        p_idx = patch_dir["patch_idx"][j]
        feature, _, _, indexNum = dataset.get_random_data(j)
        pred1, pred2 = model.predict(feature)
        pred1 = np.argmax(pred1, axis=3)
        pred2 = np.argmax(pred2, axis=3)
        pred_full_label1[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred1[0] 
        pred_full_label2[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred2[0]
        
    
    # get full feature image and mask
    full_feature , full_mask1, full_mask2 = read_img(full_img_dir.iloc[i], width=config['tiles_size'])
    
    # calculate keras MeanIOU score
    m1 = keras.metrics.MeanIoU(num_classes=config['num_classes'])
    m2 = keras.metrics.MeanIoU(num_classes=config['num_classes'])
    m1.update_state(full_mask1, pred_full_label1)
    m2.update_state(full_mask2, pred_full_label2)
    score1 = m1.result().numpy()
    score2 = m2.result().numpy()
    total_score = (score1 + score2) / 2

    # plot and saving image
        
    display({"RSLC1 AMP": full_feature[:,:,0],
            "RSLC2 AMP": full_feature[:,:,1],
            "IFG": full_feature[:,:,2],
            "RSLC1 Mask": full_mask1,
            "RSLC2 Mask": full_mask2,
            "RSLC1 (MeanIOU_{:.4f})".format(score1): pred_full_label1,
            "RSLC2 (MeanIOU_{:.4f})".format(score2): pred_full_label2
            }, i, config['prediction_val_dir'], total_score, config['experiment'])


# Model Output Path
# ----------------------------------------------------------------------------------------------

def create_paths(config, test=False, eval=False):
    """
    Summary:
        creating paths for train and test if not exists
    Arguments:
        config (dict): configuration dictionary
        test (bool): boolean variable for test directory create
    Return:
        create directories
    """
    if test:
        pathlib.Path(config['prediction_test_dir']).mkdir(
            parents=True, exist_ok=True)
    if eval:
        if config["video_path"] != 'None':
            pathlib.Path(config["dataset_dir"] + "/video_frame").mkdir(
                parents=True, exist_ok=True)
        pathlib.Path(config['prediction_eval_dir']).mkdir(
            parents=True, exist_ok=True)
    else:
        pathlib.Path(config['csv_log_dir']
                     ).mkdir(parents=True, exist_ok=True)
        pathlib.Path(config['tensorboard_log_dir']).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(config['checkpoint_dir']).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(config['prediction_val_dir']).mkdir(
            parents=True, exist_ok=True)

# Create config path
# ----------------------------------------------------------------------------------------------

# def get_config_yaml(path, args):
#     """
#     Summary:
#         parsing the config.yaml file and re organize some variables
#     Arguments:
#         path (str): config.yaml file directory
#         args (dict): dictionary of passing arguments
#     Return:
#         a dictonary
#     """
#     with open(path, "r") as f:
#         config = yaml.safe_load(f)

#     # Replace default values with passing values
#     for key in args.keys():
#         if args[key] != None:
#             config[key] = args[key]

#     if config['patchify']:
#         config['height'] = config['patch_size']
#         config['width'] = config['patch_size']

#     # Merge paths
#     config['train_dir'] = config['dataset_dir']+config['train_dir']
#     config['valid_dir'] = config['dataset_dir']+config['valid_dir']
#     config['test_dir'] = config['dataset_dir']+config['test_dir']
#     config['eval_dir'] = config['dataset_dir']+config['eval_dir']

#     config['p_train_dir'] = config['dataset_dir']+config['p_train_dir']
#     config['p_valid_dir'] = config['dataset_dir']+config['p_valid_dir']
#     config['p_test_dir'] = config['dataset_dir']+config['p_test_dir']
#     config['p_eval_dir'] = config['dataset_dir']+config['p_eval_dir']

#     # Create Callbacks paths
#     config['tensorboard_log_name'] = "{}_ex_{}_ep_{}_{}".format(
#         config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
#     config['tensorboard_log_dir'] = config['root_dir'] + \
#         '/logs/' + \
#         config['model_name']+'/'  

#     config['csv_log_name'] = "{}_ex_{}_ep_{}_{}.csv".format(
#         config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
#     config['csv_log_dir'] = config['root_dir'] + \
#         '/csv_logger/' + \
#         config['model_name']+'/'   

#     config['checkpoint_name'] = "{}_ex_{}_ep_{}_{}.hdf5".format(
#         config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
#     config['checkpoint_dir'] = config['root_dir'] + \
#         '/model/' + \
#         config['model_name']+'/'   

#     # Create save model directory
#     if config['load_model_dir'] == 'None':
#         config['load_model_dir'] = config['root_dir'] + \
#             '/model/' + \
#             config['model_name']+'/'  

#     # Create Evaluation directory
#     config['prediction_test_dir'] = config['root_dir'] + '/prediction/'+ config['model_name'] + '/test/' + config['experiment'] + '/'
#     config['prediction_eval_dir'] = config['root_dir'] + '/prediction/'+ config['model_name'] + '/eval/' + config['experiment'] + '/'
#     config['prediction_val_dir'] = config['root_dir'] + '/prediction/' + config['model_name'] + '/validation/' + config['experiment'] + '/'

#     config['visualization_dir'] = config['root_dir']+'/visualization/'

#     return config
    
    
def frame_to_video(config, fname, fps=30):
    """
    Summary:
        create video from frames
    Arguments:
        config (dict): configuration dictionary
        fname (str): name of the video
    Return:
        video
    """
    
    image_folder=config['prediction_eval_dir']
    image_names = os.listdir(image_folder)
    image_names = sorted(image_names)
    image_files = []
    for i in image_names:
        image_files.append(image_folder + "/" + i)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(fname)


# def video_to_frame(config):
#     """
#     Summary:
#         create frames from video
#     Arguments:
#         config (dict): configuration dictionary
#     Return:
#         frames
#     """
    
#     vidcap = cv2.VideoCapture(config["video_path"])
#     success,image = vidcap.read()
#     count = 0
#     while success:
#         cv2.imwrite(config['dataset_dir'] + '/video_frame' + '/frame_%06d.jpg' % count, image)     # save frame as JPEG file      
#         success,image = vidcap.read() 
#         count += 1

