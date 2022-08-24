import os
import math
import json
import rasterio
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib
from tiles import readRas
from mrc_insar_common.data import data_reader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical, Sequence
matplotlib.use('Agg')


# labels normalization values       
label_norm = {0:["_vv.tif", -17.54, 5.15],
                1:["_vh.tif",-10.68, 4.62],
                2:["_nasadem.tif",166.47, 178.47],
                3:["_jrc-gsw-change.tif", 238.76, 5.15],
                4:["_jrc-gsw-extent.tif", 2.15, 22.71],
                5:["_jrc-gsw-occurrence.tif", 6.50, 29.06],
                6:["_jrc-gsw-recurrence.tif", 10.04, 33.21],
                7:["_jrc-gsw-seasonality.tif", 2.60, 22.79],
                8:["_jrc-gsw-transitions.tif", 0.55, 1.94]}


def transform_data(inp, mask1, mask2, num_class, scale=True):
    """
    Summary:
        transform label/mask into one hot matrix and return
    Arguments:
        label (arr): label/mask
        num_classes (int): number of class in label/mask
    Return:
        one hot label matrix
    """

    scaler = MinMaxScaler((0.0,0.9999999))

    
    if scale:
        inputs = []
        for i in range(len(inp)):
            dim = inp[i].shape
            inputs.append(scaler.fit_transform(inp[i].reshape(-1, dim[-1])).reshape(dim))
    else:
        inputs = inp


    return np.array(inputs), to_categorical(mask1, num_class), to_categorical(mask2, num_class)



def read_img(data_p, in_channels=None, label=False, patch_idx=None, width=512):
    """
    Summary:
        read image with rasterio and normalize the feature
    Arguments:
        directory (str): image path to read
        in_channels (bool): number of channels to read
        label (bool): TRUE if the given directory is mask directory otherwise False
        patch_idx (list): patch indices to read
    Return:
        numpy.array
    """

    rslc0 = data_reader.readBin(data_p["rslc0"], width=width, dataType='floatComplex') # SLC(RSLC) image
    rslc1 = data_reader.readBin(data_p["rslc1"], width=width, dataType='floatComplex') # SLC(RSLC) image
    rslc0_label = readRas(data_p["rslc0_label"])[0] # SLC(RSLC) label
    rslc1_label = readRas(data_p["rslc1_label"])[0] # SLC(RSLC) label

    rslc0_label[rslc0_label==255] = 1
    rslc1_label[rslc1_label==255] = 1

    rslc0_amp = np.abs(rslc0)
    rslc1_amp = np.abs(rslc1)
    ifg = np.angle(rslc1*np.conjugate(rslc0))
    if patch_idx:
        rslc0_amp = rslc0_amp[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]]
        rslc1_amp = rslc1_amp[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]]
        ifg = ifg[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]]
        inputs = np.stack([rslc0_amp, rslc1_amp, ifg], axis=-1)
        
        rslc0_label = rslc0_label[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]]
        rslc1_label = rslc1_label[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]]
    else:
        inputs = np.stack([rslc0_amp, rslc1_amp, ifg], axis=-1)

    

    return inputs, rslc0_label, rslc1_label


def data_split(images, config):
    """
    Summary:
        split dataset into train, valid and test
    Arguments:
        images (list): all image directory list
        masks (list): all mask directory
        config (dict): Configuration directory
    Return:
        return the split data.
    """


    x_train, x_rem = train_test_split(images, train_size = config['train_size'], random_state=42)
    x_valid, x_test = train_test_split(x_rem, test_size = 0.5, random_state=42)
    return x_train, x_valid, x_test


def save_csv(dictionary, config, name):
    """
    Summary:
        save csv file
    Arguments:
        dictionary (dict): data as a dictionary object
        config (dict): Configuration directory
        name (str): file name to save
    Return:
        save file
    """
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv((config['dataset_dir']+name), index=False, header=True)


def data_path_split(config):
    
    """
    Summary:
        spliting data into train, test, valid
    Arguments:
        config (dict): Configuration directory
    Return:
        save file
    """

    paths = pd.read_csv((config['dataset_dir']+"tiles_path.csv"))
    train, valid, test = data_split(paths, config)

    save_csv(train, config, "train.csv")
    save_csv(valid, config, "valid.csv")
    save_csv(test, config, "test.csv")


def class_percentage_check(label):
    
    """
    Summary:
        check class percentage of a single mask image
    Arguments:
        label (numpy.ndarray): mask image array
    Return:
        dict object holding percentage of each class
    """
    
    total_pix = label.shape[0]*label.shape[0]
    class_one = np.sum(label)
    class_zero_p = total_pix-class_one
    return {"zero_class":((class_zero_p/total_pix)*100),
            "one_class":((class_one/total_pix)*100)
    }



def save_patch_idx(path, patch_size=256, stride=8, test=None, patch_class_balance=None):
    """
    Summary:
        finding patch image indices for single image based on class percentage. work like convolutional layer
    Arguments:
        path (str): image path
        patch_size (int): size of the patch image
        stride (int): how many stride to take for each patch image
    Return:
        list holding all the patch image indices for a image
    """
    
    img = readRas(path)[0] # SLC(RSLC) label
    img[img==255] = 1
    
    # calculating number patch for given image
    patch_height = int((img.shape[0]-patch_size)/stride)+1 # [{(image height-patch_size)/stride}+1]
    patch_weight = int((img.shape[1]-patch_size)/stride)+1 # [{(image weight-patch_size)/stride}+1]
    
    # total patch images = patch_height * patch_weight
    patch_idx = []
    
    # image column traverse
    for i in range(patch_height):
        s_row = i*stride
        e_row = s_row+patch_size
        if e_row <= img.shape[0]:
            
            # image row traverse
            for j in range(patch_weight):
                start = (j*stride)
                end = start+patch_size
                if end <= img.shape[1]:
                    tmp = img[s_row:e_row, start:end]
                    percen = class_percentage_check(tmp) # find class percentage
                    
                    # take all patch for test images
                    if patch_class_balance or test=='test':
                        patch_idx.append([s_row, e_row, start, end])
                    
                    # store patch image indices based on class percentage
                    else:
                        if percen["one_class"]>2.0:
                            patch_idx.append([s_row, e_row, start, end])
    return  patch_idx


def write_json(target_path, target_file, data):
    """
    Summary:
        save dict object into json file
    Arguments:
        target_path (str): path to save json file
        target_file (str): file name to save
        data (dict): dictionary object holding data
    Returns:
        save json file
    """
    
    
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)


def patch_images(data, config, name):
    """
    Summary:
        save all patch indices of all images
    Arguments:
        data: data file contain image paths
        config (dict): configuration directory
        name (str): file name to save patch indices
    Returns:
        save patch indices into file
    """
    temp = {}
    for i in data.columns:
        temp[i] = []
    
    temp["patch_idx"] = []
    
    # loop through all images
    for i in range(len(data)):
        
        # fetching patch indices
        patches = save_patch_idx(data.rslc0_label.values[i], patch_size=config['patch_size'], stride=config['stride'], test=name.split("_")[0], patch_class_balance=config['patch_class_balance'])
        
        # generate data point for each patch image
        for patch in patches:
            for col in data.columns:
                temp[col].append(data[col].values[i])
            temp["patch_idx"].append(patch)
    
    # save data
    write_json((config['dataset_dir']+"json/"), (name+str(config['patch_size'])+'.json'), temp)

# Data Augment class
# ----------------------------------------------------------------------------------------------
class Augment:
    def __init__(self, batch_size, channels, ratio=0.3, seed=42):
        super().__init__()
        """
        Summary:
            initialize class variables
        Arguments:
            batch_size (int): how many data to pass in a single step
            ratio (float): percentage of augment data in a single batch
            seed (int): both use the same seed, so they'll make the same random changes.
        Return:
            class object
        """


        self.ratio=ratio
        self.channels= channels
        self.aug_img_batch = math.ceil(batch_size*ratio)
        self.aug = A.Compose([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Blur(p=0.5),])

    def call(self, data, patch_idx=None):
        """
        Summary:
            randomly select a directory and augment data 
            from that specific image and mask
        Arguments:
            feature_dir (list): all train image directory list
            label_dir (list): all train mask directory list
        Return:
            augmented image and mask
        """

        # choose random image from dataset to augment
        aug_idx = np.random.randint(0, len(data), self.aug_img_batch)
        features = []
        labels = []
        labels2 = []

        for i in aug_idx:
            if patch_idx:
                data_p = read_img(data.iloc[i], in_channels = self.channels, patch_idx=patch_idx[i])
            else:
                data_p = read_img(data.iloc[i], in_channels = self.channels)

            masks = np.stack([data_p[1], data_p[2]], axis=-1)
            augmented = self.aug(image=data_p[0], mask=masks)
            features.append(augmented['image'])
            labels.append(augmented['mask'][:,:,0])
            labels2.append(augmented['mask'][:,:,1])
        return features, labels, labels2



# Dataloader class
# ----------------------------------------------------------------------------------------------

class MyDataset(Sequence):

    def __init__(self, data, in_channels, 
                 batch_size, num_class, patchify,transform_fn=None,
                 augment=None, weights=None, patch_idx=None, tile_width=512):
        """
        Summary:
            initialize class variables
        Arguments:
            img_dir (list): all image directory
            tgt_dir (list): all mask/ label directory
            in_channels (int): number of input channels
            batch_size (int): how many data to pass in a single step
            patchify (bool): set TRUE if patchify experiment
            transform_fn (function): function to transform mask images for training
            num_class (int): number of class in mask image
            augment (object): Augment class object
            weight (list): class weight for imblance class
            patch_idx (list): list of patch indices
        Return:
            class object
        """


        self.data = data
        self.patch_idx = patch_idx
        self.patchify = patchify
        self.in_channels = in_channels
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights
        self.tile_width = tile_width



    def __len__(self):
        """
        return total number of batch to travel full dataset
        """


        return math.ceil(len(self.data) // self.batch_size)


    def __getitem__(self, idx):
        """
        Summary:
            create a single batch for training
        Arguments:
            idx (int): sequential batch number
        Return:
            images and masks as numpy array for a single batch
        """


        # get index for single batch
        batch = self.data[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        
        if self.patchify:
            batch_patch = batch["patch_idx"].values
        
        inputs = []
        masks = []
        masks2 = []

        for i in range(len(batch)):
            if self.patchify:
                data_p = read_img(batch.iloc[i], in_channels = self.in_channels, patch_idx=batch_patch[i], width=self.tile_width)
            else:
                data_p = read_img(batch.iloc[i], in_channels = self.in_channels, width=self.tile_width)
            
            inputs.append(data_p[0])
            masks.append(data_p[1])
            masks2.append(data_p[2])
        
        # augment data using Augment class above if augment is true
        if self.augment:
            if self.patchify:
                aug_imgs, aug_masks1, aug_masks2 = self.augment.call(self.data, self.patch_idx) # augment images and mask randomly
            else:
                aug_imgs, aug_masks1, aug_masks2 = self.augment.call(self.data) # augment images and mask randomly
            inputs = inputs + aug_imgs
            masks = masks + aug_masks1
            masks2 = masks2 + aug_masks2
        
        inputs, masks, masks2 = self.transform_fn(inputs, masks, masks2, self.num_class)

        if self.weights != None:

            class_weights = tf.constant(self.weights)
            class_weights = class_weights/tf.reduce_sum(class_weights)
            y_weights = tf.gather(class_weights, indices=tf.cast(masks, tf.int32))#([self.paths[i] for i in indexes])
            y_weights2 = tf.gather(class_weights, indices=tf.cast(masks2, tf.int32))

            return tf.convert_to_tensor(inputs), [y_weights, y_weights2]

        return inputs, [masks, masks2]
        # return {"input_1": inputs,
        # "out1": masks,
        # "out2": masks2}
    

    def get_random_data(self, idx=-1):
        """
        Summary:
            randomly chose an image and mask or the given index image and mask
        Arguments:
            idx (int): specific image index default -1 for random
        Return:
            image and mask as numpy array
        """



        if idx!=-1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.data))
        
        inputs = []
        masks = []
        masks2 = []
        if self.patchify:
            data_p = read_img(self.data.iloc[idx], in_channels = self.in_channels, patch_idx=self.data.patch_idx[idx], width=self.tile_width)
        else:
            data_p = read_img(self.data.iloc[idx], in_channels = self.in_channels, width=self.tile_width)
            
        inputs.append(data_p[0])
        masks.append(data_p[1])
        masks2.append(data_p[2])
        
        inputs, masks, masks2 = self.transform_fn(inputs, masks, masks2, self.num_class)

        if self.weights != None:

            class_weights = tf.constant(self.weights)
            class_weights = class_weights/tf.reduce_sum(class_weights)
            y_weights = tf.gather(class_weights, indices=tf.cast(masks, tf.int32))#([self.paths[i] for i in indexes])
            y_weights2 = tf.gather(class_weights, indices=tf.cast(masks2, tf.int32))

            return tf.convert_to_tensor(inputs), y_weights, y_weights2, idx

        return tf.convert_to_tensor(inputs), tf.convert_to_tensor(masks), tf.convert_to_tensor(masks2), idx



def get_train_val_dataloader(config):
    """
    Summary:
        read train and valid image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        train and valid dataloader
    """


    if not (os.path.exists(config['train_dir'])):
        data_path_split(config)
    
    if not (os.path.exists(config["p_train_dir"])) and config['patchify']:
        print("Saving patchify indices for train and test.....")
        train_data = pd.read_csv(config['train_dir'])
        valid_data = pd.read_csv(config['valid_dir'])
        
        if config["patch_class_balance"]:
            patch_images(train_data, config, "train_patch_WOC_")
            patch_images(valid_data, config, "valid_patch_WOC_")
        else:
            patch_images(train_data, config, "train_patch_")
            patch_images(valid_data, config, "valid_patch_")

        
    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config['p_train_dir'], 'r') as j:
            train_dir = json.loads(j.read())
        with open(config['p_valid_dir'], 'r') as j:
            valid_dir = json.loads(j.read())
        train_idx = train_dir['patch_idx']
        valid_idx = valid_dir['patch_idx']
        train_dir = pd.DataFrame.from_dict(train_dir)
        valid_dir = pd.DataFrame.from_dict(valid_dir)

    else:
        print("Loading features and masks directories.....")
        train_dir = pd.read_csv(config['train_dir'])
        valid_dir = pd.read_csv(config['valid_dir'])
        train_idx = None
        valid_idx = None

    print("train Example : {}".format(len(train_dir)))
    print("valid Example : {}".format(len(valid_dir)))


    # create Augment object if augment is true
    if config['augment'] and config['batch_size']>1:
        augment_obj = Augment(config['batch_size'], config['in_channels'])
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch # new batch size after augment data for train
    else:
        n_batch_size = config['batch_size']
        augment_obj = None

    # class weight
    if config['weights']:
        weights=tf.constant(config['balance_weights'])
    else:
        weights = None
    
    # create dataloader object
    train_dataset = MyDataset(train_dir,
                                in_channels=config['in_channels'], patchify=config['patchify'],
                                batch_size=n_batch_size, transform_fn=transform_data, 
                                num_class=config['num_classes'], augment=augment_obj, 
                                weights=weights, patch_idx=train_idx, tile_width=config["tile_width"])

    val_dataset = MyDataset(valid_dir,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=valid_idx, tile_width=config["tile_width"])
    
    return train_dataset, val_dataset


def get_test_dataloader(config):
    """
    Summary:
        read test image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        test dataloader
    """


    if not (os.path.exists(config['test_dir'])):
        data_path_split(config)
    
    if not (os.path.exists(config["p_test_dir"])) and config['patchify']:
        print("Saving patchify indices for test.....")
        data = pd.read_csv(config['test_dir'])
        patch_images(data, config, "test_patch_")
        
    
    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config['p_test_dir'], 'r') as j:
            test_dir = json.loads(j.read())
        test_features = test_dir['feature_ids']
        test_masks = test_dir['masks']
        test_idx = test_dir['patch_idx']
    
    else:
        print("Loading features and masks directories.....")
        test_dir = pd.read_csv(config['test_dir'])
        test_features = test_dir.feature_ids.values
        test_masks = test_dir.masks.values
        test_idx = None

    print("test Example : {}".format(len(test_features)))


    test_dataset = MyDataset(test_features,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=test_idx)
    
    return test_dataset