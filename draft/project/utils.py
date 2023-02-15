import os
import json
import math
import yaml
import glob
import numpy as np
import pandas as pd
import pathlib
from loss import *
import tensorflow as tf
import earthpy.plot as ep
import earthpy.spatial as es
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
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
        lr = self.config['learning_rate'] * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
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
        if (epoch % self.config['val_plot_epoch'] == 0): # every after certain epochs the model will predict mask
            show_predictions(self.val_dataset, self.model, self.config, True)

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
        if self.config['csv']:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(self.config['csv_log_dir'], self.config['csv_log_name']), separator = ",", append = False))
        
        if self.config['checkpoint']:
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(self.config['checkpoint_dir'], self.config['checkpoint_name']), save_best_only = True))
        
        if self.config['tensorboard']:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir = os.path.join(self.config['tensorboard_log_dir'], self.config['tensorboard_log_name'])))
        
        if self.config['lr']:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(schedule = self.lr_scheduler))
        
        if self.config['early_stop']:
            self.callbacks.append(keras.callbacks.EarlyStopping(monitor = 'my_mean_iou', patience = self.config['patience']))
        
        if self.config['val_pred_plot']:
            self.callbacks.append(SelectCallbacks(val_dataset, model, self.config))
        
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

def display(display_list, idx, directory, exp):
    """
    Summary:
        save all images into single figure
    Arguments:
        display_list (dict): a python dictionary key is the title of the figure
        idx (int) : image index in dataset object
        directory (str) : path to save the plot figure
        score (float) : accuracy of the predicted mask
        exp (str): experiment name
    Return:
        save images figure into directory
    """
    # scaler = MinMaxScaler((0.0,0.9999999))
    plt.figure(figsize=(12, 20))
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
            plt.title(title[i], fontsize=6)
            c2 = plt.imshow((display_list[title[i]])**0.3, cmap="gray", interpolation=None)
            ax = plt.gca()
            fig = plt.gcf()
            cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
            cbar.ax.tick_params(labelsize=6) 
            plt.axis('off')
            
        # ifr plot
        elif title[i]=="ifg":
            plt.title(title[i], fontsize=6)
            c2 = plt.imshow((display_list[title[i]]), cmap="jet", interpolation=None)
            ax = plt.gca()
            fig = plt.gcf()
            cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
            cbar.ax.tick_params(labelsize=6) 
            plt.axis('off')

        # plot labels
        else:
            plt.title(title[i], fontsize=6)
            c2 = plt.imshow((display_list[title[i]]), vmin=0, vmax=1)
            ax = plt.gca()
            fig = plt.gcf()
            cbar = fig.colorbar(c2, ax=ax, fraction=0.046)
            cbar.ax.tick_params(labelsize=6) 
            plt.axis('off')

    prediction_name = "img_ex_{}_{}.png".format(exp, idx) # create file name to save
    plt.savefig(os.path.join(directory, prediction_name), bbox_inches='tight', dpi=800)
    plt.clf()
    plt.cla()
    plt.close()


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
                      }, idx, directory, config['experiment'])
            idx += 1

# Combine patch images and save
# ----------------------------------------------------------------------------------------------
def patch_show_predictions(dataset, model, config):
    # predict patch images and merge together 
    
    with open(config['p_test_dir'], 'r') as j:
        patch_test_dir = json.loads(j.read())
    
    df = pd.DataFrame.from_dict(patch_test_dir)
    test_dir = pd.read_csv(config['test_dir'])
    total_score = 0.0
    
    # loop to traverse full dataset
    for i in range(len(test_dir)):
        idx = df[df["masks"]==test_dir["masks"][i]].index
        
        # construct a single full image from prediction patch images
        pred_full_label = np.zeros((512,512), dtype=int)
        for j in idx:
            p_idx = patch_test_dir["patch_idx"][j]
            feature, mask, _ = dataset.get_random_data(j)
            pred_mask = model.predict(feature)
            pred_mask = np.argmax(pred_mask, axis = 3)
            pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred_mask[0]
        
        
        # read original image and mask
        feature = read_img(test_dir["feature_ids"][i], in_channels=config['in_channels'])
        mask = transform_data(read_img(test_dir["masks"][i], label=True), config['num_classes'])
        
        # calculate keras MeanIOU score
        m = keras.metrics.MeanIoU(num_classes=config['num_classes'])
        m.update_state(np.argmax([mask], axis = 3), [pred_full_label])
        score = m.result().numpy()
        total_score += score
        
        # plot and saving image
        display({"VV": feature[:,:,0],
                    "VH": feature[:,:,1],
                    "DEM": feature[:,:,2],
                    "Mask": np.argmax([mask], axis = 3)[0],
                    "Prediction (MeanIOU_{:.4f})".format(score): pred_full_label
                    }, i, config['prediction_test_dir'], score, config['experiment'])


# GPU setting
# ----------------------------------------------------------------------------------------------
def set_gpu(gpus):
    """
    Summary:
        setting multi-GPUs or single-GPU strategy for training
    Arguments:
        gpus (str): comma separated str variable i.e. "0,1,2"
    Return:
        gpu strategy object
    """
    gpus = gpus.split(",")
    if len(gpus)>1:
        print("MirroredStrategy Enable")
        GPUS = []
        for i in range(len(gpus)):
            GPUS.append("GPU:{}".format(gpus[i]))
        strategy = tf.distribute.MirroredStrategy(GPUS)
    else:
        print("OneDeviceStrategy Enable")
        GPUS = []
        for i in range(len(gpus)):
            GPUS.append("GPU:{}".format(gpus[i]))
        strategy = tf.distribute.OneDeviceStrategy(GPUS[0])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    
    return strategy

# Model Output Path
# ----------------------------------------------------------------------------------------------
def create_paths(config, test=False):
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
        pathlib.Path(config['prediction_test_dir']).mkdir(parents = True, exist_ok = True)
    else:
        pathlib.Path(config['csv_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['tensorboard_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['checkpoint_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['prediction_val_dir']).mkdir(parents = True, exist_ok = True)

# Create config path
# ----------------------------------------------------------------------------------------------
def get_config_yaml(path, args):
    """
    Summary:
        parsing the config.yaml file and re organize some variables
    Arguments:
        path (str): config.yaml file directory
        args (dict): dictionary of passing arguments
    Return:
        a dictonary
    """
    with open(path, "r") as f:
      config = yaml.safe_load(f)
    
    # Replace default values with passing values
    for key in args.keys():
        if args[key] != None:
            config[key] = args[key]
            
    config['height'] = config['patch_size']
    config['width'] = config['patch_size']
    
    # Merge paths
    config['train_dir'] = config['dataset_dir']+config['train_dir']
    config['valid_dir'] = config['dataset_dir']+config['valid_dir']
    config['test_dir'] = config['dataset_dir']+config['test_dir']
    
    config['p_train_dir'] = config['dataset_dir']+config['p_train_dir']
    config['p_valid_dir'] = config['dataset_dir']+config['p_valid_dir']
    config['p_test_dir'] = config['dataset_dir']+config['p_test_dir']
    
    # Create Callbacks paths
    config['tensorboard_log_name'] = "{}_ex_{}_epochs_{}_{}".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir']+'/logs/'+config['model_name']+'/'

    config['csv_log_name'] = "{}_ex_{}_epochs_{}_{}.csv".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir']+'/csv_logger/'+config['model_name']+'/'

    config['checkpoint_name'] = "{}_ex_{}_epochs_{}_{}.hdf5".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'

    # Create save model directory
    if config['load_model_dir']=='None':
        config['load_model_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'
    
    # Create Evaluation directory
    config['prediction_test_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/test/'
    config['prediction_val_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/validation/'
    
    config['visualization_dir'] = config['root_dir']+'/visualization/'

    return config



# Helper functions for visualizing Sentinel-1 images
def scale_img(matrix):
    """
    Returns a scaled (H, W, D) image that is visually inspectable.
    Image is linearly scaled between min_ and max_value, by channel.

    Args:
        matrix (np.array): (H, W, D) image to be scaled

    Returns:
        np.array: Image (H, W, 3) ready for visualization
    """
    # Set min/max values
    min_values = np.array([[-23, -28, 0.2]])
    max_values = np.array([[0, -5, 1]])

    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    # Scale by min/max
    matrix = (matrix - min_values) / (
        max_values - min_values
    )
    matrix = np.reshape(matrix, [w, h, d])

    # Limit values to 0/1 interval
    return matrix.clip(0, 1)

def create_false_color_composite(vv_img, vh_img):
    """
    Returns a S1 false color composite for visualization.

    Args:
        path_vv (str): path to the VV band
        path_vh (str): path to the VH band

    Returns:
        np.array: image (H, W, 3) ready for visualization
    """    
    # Stack arrays along the last dimension
    s1_img = np.stack((vv_img, vh_img), axis=-1)

    # Create false color composite
    img = np.zeros((512, 512, 3), dtype=np.float32)
    img[:,:,:2] = s1_img.copy()
    img[:, :, 2] = (s1_img[:, :, 0]*s1_img[:, :, 1])


    return scale_img(img)

def plot_3d():
    
    # extract csv logger paths
    paths = glob.glob("/home/mdsamiul/github_project/flood_water_mapping_segmentation/csv_logger/ad_unet/*.csv")
    
    # smooth plotting values
    def my_tb_smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
        """

        ref: https://stackoverflow.com/questions/42011419/is-it-possible-to-call-tensorboard-smooth-function-manually

        :param scalars:
        :param weight:
        :return:
        """
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed: list = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
        return smoothed
    
    # initialize variables
    epoch = []
    mean_iou = []
    patch = []
    
    # read data and smooth for plot
    for path in paths:
        if path.split("_")[-5] == "patchify" and path.split("_")[-2]=="60":
            patch_size = int(path.split("_")[-4])
        else:
            patch_size = 512
        data = pd.read_csv(path)
        epoch.append(range(0, 60*2))
        mean_iou.append(my_tb_smooth(data["val_my_mean_iou"][:60], 0.95)+(my_tb_smooth(data["val_my_mean_iou"][:60], 0.95)[::-1]))
        patch.append(([patch_size]*60)+([patch_size]*60))
    
    # numpy array convert for plot
    epoch = np.array(epoch)
    mean_iou = np.array(mean_iou)
    patch = np.array(patch)
    
    # create figure and plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(patch, epoch, mean_iou,rstride=1, cstride=1,
                    cmap='jet', edgecolor='none')\
    
    # customize figure visualization with labels and title
    ax.set_yticklabels([-1,10, 20, 40, 40,20, 10])
    ax.view_init(elev=20, azim=70)
    plt.title('MeanIou accuracy for different patch size')
    ax.set_zlabel("MeanIou accuracy")
    ax.set_ylabel('Epoch')
    ax.set_xlabel('Patch')
    plt.savefig("area2.png", dpi=800)
    plt.show()

def find_best_worst():

    path = glob.glob("/home/mdsamiul/github_project/flood_water_mapping_segmentation/prediction/mnet/test/*.*")

    scores = np.zeros((55, 4), dtype=np.float32)
    for i in range(len(path)):
        id = int(path[i].split("_")[-3])
        acu = np.float32(path[i].split("_")[-1].replace(".png", ""))
        if path[i].split("_")[-4] == "patchify":
            scores[id][2] = acu
        elif path[i].split("_")[-4] == "balance":
            scores[id][1] = acu
        elif path[i].split("_")[-4] == "WOC":
            scores[id][3] = acu
        else:
            scores[id][0] = acu

    df = pd.DataFrame(scores)
    df.to_csv("predic_score.csv")
