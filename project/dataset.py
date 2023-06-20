import os
import math
import json
import glob
import rasterio
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib
from progressbar import ProgressBar
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical, Sequence
from mrc_insar_common.data import data_reader

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

rasMagic = 0x59A66A95
rasTypeH = '>i4'
rasTypeA = '>u1'
dtypes = {
    'byte': '>u1',
    'int': '>i2',
    'uint': '>u2',
    'long': '>i4',
    'ulong': '>u4',
    'float': '>f4',
    'double': '>f8',
    'scomplex': '>i2',
    'fcomplex': '>c8',
    'complex': '>c8',
    'bool': 'bool',
    'ptsel_inds': '<i4'
}

def transform_data(inp1, inp2, mask1, mask2, num_class, scale=True):
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

    
    # if scale:     prev
    #     inputs = []
    #     for i in range(len(inp)):
    #         dim = inp[i].shape
    #         inputs.append(scaler.fit_transform(inp[i].reshape(-1, dim[-1])).reshape(dim))
    # else:
    #     inputs = inp


    # return np.array(inputs), to_categorical(mask1, num_class), to_categorical(mask2, num_class)
    

    if scale:
        inputs1 = []
        inputs2 = []
        for i in range(len(inp1)):
            inputs1.append(scaler.fit_transform(inp1[i]))
            inputs2.append(scaler.fit_transform(inp2[i]))
    else:
        inputs1 = inp1
        inputs2 = inp2


    return np.array(inputs1), np.array(inputs2), to_categorical(mask1, num_class), to_categorical(mask2, num_class)


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
        # inputs = np.stack([rslc0_amp, rslc1_amp], axis=-1)

    
    # return inputs, rslc0_label, rslc1_label           # prev
    return rslc0_amp, rslc1_amp, rslc0_label, rslc1_label


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
    # converting dictionary to pandas dataframe
    df = pd.DataFrame.from_dict(dictionary)
    # from dataframe to csv
    df.to_csv((config['dataset_dir']+name), index=False, header=True)


def video_to_frame(config):
    """
    Summary:
        create frames from video
    Arguments:
        config (dict): configuration dictionary
    Return:
        frames
    """
    
    vidcap = cv2.VideoCapture(config["video_path"])
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(config['dataset_dir'] + '/video_frame' + '/frame_%06d.jpg' % count, image)     # save frame as JPEG file      
        success,image = vidcap.read() 
        count += 1

def colorspace(image):
    """
    Swap the B and R color channels of a truecolor image.
    Use this if ras_type is RT_FORMAT_RGB (3), since 24-bit images
    are saved by default using RT_STANDARD (1) which assumes BGR

    @param image: The image array in RGB format
    :type image:  Numpy Array (height, width, 3)

    :rtype:  Numpy Array (height, width, 3)
    :return: The new image array in BGR format
    """

    if image.ndim == 3 and image.shape[2] == 3:
        return image[..., ::-1]


def writeRas(image, filename, cmap=None):
    """
    Saves a numpy array as a Sun Raster image.
    Only 8/24-bit images are supported.
 
    @param image: The array to save to disk
    :type image:  Numpy Array
 
    @param filename: Path of raster file
    :type filename:  String
 
    @param cmap: Save this 3x256 colormap for 8-bit images
    :type cmap:  Numpy Array
 
    :raises: IOError, TypeError
    """
 
    if image.ndim not in (2, 3):
        raise TypeError("Only 8-bit and 24-bit rasters are supported")
 
    height = image.shape[0]
    width = image.shape[1]
    padding = (width % 2 != 0)
 
    depth = 8 if (image.ndim == 2) else 24
    cmap = empty_raster_cmap() if (cmap is None and depth == 8) else cmap
 
    image = image.reshape(height, -1).astype(rasTypeA)
    image = np.hstack((image, np.zeros((height, 1), dtype=rasTypeA))) if padding else image
 
    header = np.empty((8, 1), dtype=rasTypeH)
    header[0] = rasMagic
    header[1] = width
    header[2] = height
    header[3] = depth
    header[4] = image.nbytes if image.nbytes < np.iinfo(rasTypeH).max else 0
    header[5] = 1
    header[6] = 0 if depth == 24 else 1
    header[7] = 0 if depth == 24 else cmap.nbytes
 
    outfile = open(filename, 'wb')
    header.tofile(outfile)
    if cmap is not None:
        cmap.tofile(outfile)
    image.tofile(outfile)
    outfile.close()
 

def empty_raster_cmap():
    cmap = np.empty((3, 256), dtype=np.uint8)
    cmap[:, ] = np.arange(256)
    return cmap

def readRas(filename, crop=None, mmap=False):
    """
    Reads a Sun Raster image into a numpy array.
    Only 8/24-bit images are supported.

    @param filename: Path of raster file
    :type filename:  String

    @param crop: Crop coordinates X Offset, Y Offset, Width, Height
    :type crop: List of 4 Integers

    :rtype:  tuple of ndarray
    :return: (data, cmap) or (data, None)

    :raises: IOError, TypeError
    """

    infile = open(filename, 'rb')
    header = np.fromfile(infile, dtype=rasTypeH, count=8)

    magic_num = header[0]
    width = header[1]
    height = header[2]
    depth = header[3]
    ras_type = header[5]
    cmap_type = header[6]
    cmap_len = header[7]

    padding = (width % 2 != 0)
    bpp = depth // 8
    length = width * height * bpp
    length += height if padding else 0

    if magic_num != rasMagic:
        raise TypeError("Input is not a valid Sun Raster Image")
    if depth not in (8, 24):
        raise TypeError("Only 8-bit and 24-bit rasters are supported")
    if cmap_type not in (0, 1):
        raise TypeError("Unsupported color map type")

    if mmap:
        if cmap_len:
            cmap_raw = np.memmap(infile, dtype=rasTypeA, mode='r', offset=4 * 8 * np.dtype(rasTypeA).itemsize)
            cmap = cmap_raw[:cmap_len].reshape(3, -1)
        else:
            cmap = None
    else:
        cmap = np.fromfile(infile, dtype=rasTypeA, count=cmap_len).reshape(3, -1) if cmap_len else None

    # Handle cropping
    if crop is not None:
        cx, cy, cw, ch = list(map(int, crop))
        if (cx + cw) > width or (cy + ch) > height or cx < 0 or cy < 0 or cw < 0 or ch < 0:
            raise IOError("The specified crop coordinates are invalid")

        height = ch
        pad_len = ch if padding else 0
        length = (ch * width * bpp) + pad_len
        pad_len = cy if padding else 0
        infile.seek((cy * width * bpp) + pad_len, os.SEEK_CUR)

    if mmap:
        if cmap_len:
            offset = 4 * 8 * np.dtype(rasTypeA).itemsize + cmap_len
        else:
            offset = 4 * 8 * np.dtype(rasTypeA).itemsize
        image_raw = np.memmap(infile, dtype=rasTypeA, mode='r', offset=offset)
        image = image_raw[:length].reshape(height, -1)
    else:
        image = np.fromfile(infile, dtype=rasTypeA, count=length).reshape(height, -1)

    if padding:
        image = image[..., :-1]

    if depth == 24:
        image = image.reshape(height, -1, 3)
        if ras_type == 3:
            image = colorspace(image)

    if crop:
        image = image[:, cx:(cx + cw)]

    infile.close()
    return image, cmap

def save_tiles(path, tiles_size, stride, width, height, out_path=None, label=False):
    """
    Summary:
        create overlap or non-overlap tiles from large rslc image
    Arguments:
        path (str): image path
        patch_size (int): size of the patch image
        stride (int): how many stride to take for each patch image
    Return:
        list holding all the patch image indices for a image
    """
    if label:
        slc, cmap = readRas(path)
    else:
        slc = data_reader.readBin(path, width=width, dataType='floatComplex') # SLC(RSLC) image
    
    
    pathlib.Path(out_path).mkdir(parents = True, exist_ok = True)
    slc_id = path.split("/")[-1].split(".")[0].split("\\")[-1]
    
    # calculating number patch for given image
    patch_height = int((slc.shape[0]-tiles_size)/stride) + 1 # [{(image height-patch_size)/stride}+1]
    patch_width = int((slc.shape[1]-tiles_size)/stride) + 1 # [{(image width-patch_size)/stride}+1]
    

    pbar = ProgressBar()
    # image column traverse
    for i in pbar(range(patch_height)):
        s_row = i*stride
        e_row = s_row + tiles_size
        for j in range(patch_width):
            start = (j * stride)
            end = start + tiles_size
            tmp = slc[s_row:e_row, start:end]
            if label:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(j)+".ras"
                writeRas(tmp, f_name, cmap)
            else:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(j)+".rslc"
                data_reader.writeBin(f_name, tmp, "floatComplex")
    
    # if shape does not match with patch multiplication
    if (patch_height*tiles_size)<slc.shape[0]:
        s_row = slc.shape[0]-tiles_size
        e_row = slc.shape[0]
        for j in range(patch_width):
            start = (j*stride)
            end = start+tiles_size
            tmp = slc[s_row:e_row, start:end]
            if label:
                f_name = out_path+slc_id+"_"+str(patch_height)+"_"+str(j)+".ras"
                writeRas(tmp, f_name, cmap)
            else:
                f_name = out_path+slc_id+"_"+str(patch_height)+"_"+str(j)+".rslc"
                data_reader.writeBin(f_name, tmp, "floatComplex")
    
    if (patch_width*tiles_size)<slc.shape[1]:
        for i in range(patch_height):
            s_row = i*stride
            e_row = s_row+tiles_size
            start = slc.shape[1]-tiles_size
            end = slc.shape[1]
            tmp = slc[s_row:e_row, start:end]
            if label:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(patch_width)+".ras"
                writeRas(tmp, f_name, cmap)
            else:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(patch_width)+".rslc"
                data_reader.writeBin(f_name, tmp, "floatComplex")
        
def data_path_split(config):
    """
    Summary:
        spliting data into train, test, valid
    Arguments:
        config (dict): Configuration directory
    Return:
        save file
    """

    if not (os.path.exists((config['dataset_dir']+"tiles_path.csv"))):
        
        print("Creating tiles: ")
        wid, hgt = config['actual_width'], config['actual_height']  # from the masterpar.xml
        tiles_size = config['tiles_size']
        
        rslcs = sorted(glob.glob(config["dataset_dir"]+"rslc/*.rslc.notopo"))
        label = sorted(glob.glob(config["dataset_dir"]+"label/*.ras"))

        data_paths = {}

        assert len(rslcs)== len(label), "Invalid number of label or rslcs" # must have same number of rslc and label

        for i in range(len(rslcs)-1):
            out_path = config["dataset_dir"]+"rslc"+str(i)+"/"
            save_tiles(rslcs[i], tiles_size=tiles_size, stride=tiles_size, width=wid, height=hgt, out_path=(out_path+"features/"))
            save_tiles(label[i], tiles_size=tiles_size, stride=tiles_size, width=wid, height=hgt, out_path=(out_path+"label/"), label=True)
            data_paths["rslc"+str(i)] = sorted(glob.glob((out_path+"features/*.rslc")))
            data_paths["rslc"+str(i)+"_label"] = sorted(glob.glob((out_path+"label/*.ras")))
    
        df = pd.DataFrame(data_paths)
        df.to_csv((config["dataset_dir"]+"tiles_path.csv"), index=False)
    
    paths = pd.read_csv((config['dataset_dir']+"tiles_path.csv"))
    train, valid, test = data_split(paths, config)

    save_csv(train, config, "train.csv")
    save_csv(valid, config, "valid.csv")
    save_csv(test, config, "test.csv")
    
    
    
def eval_data_path_split(config):
    """
    Summary:
        for evaltion generate frame from video if video path is given and create csv file from testing folder
    Arguments:
        config (dict): Configuration directory
    Return:
        csv file
    """
    
    data_path = config["dataset_dir"]
    images = []
    
    # video is provided then it will generate frame from video
    if config["video_path"] != 'None':
        video_to_frame(config)
        image_path = data_path + "/video_frame"
        
    else:
        image_path = data_path
        
    image_names = os.listdir(image_path)
    image_names = sorted(image_names)
    
    for i in image_names:
        images.append(image_path + i)       # + "/" 

    # creating dictionary for train, test and validation
    eval = {'feature_ids': images, 'masks': images}

    # saving dictionary as csv files
    save_csv(eval, config, "eval.csv")
    

def class_percentage_check(label):
    """
    Summary:
        check class percentage of a single mask image
    Arguments:
        label (numpy.ndarray): mask image array
    Return:
        dict object holding percentage of each class
    """
    # calculating total pixels
    total_pix = label.shape[0]*label.shape[0]
    # get the total number of pixel labeled as 1
    class_one = np.sum(label)
    # get the total number of pixel labeled as 0
    class_zero_p = total_pix-class_one
    # return the pixel percent of each class
    return {"zero_class": ((class_zero_p/total_pix)*100),
            "one_class": ((class_one/total_pix)*100)
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
    # read the image
    img = readRas(path)[0] # SLC(RSLC) label
    img[img==255] = 1

    # calculating number patch for given image
    # [{(image height-patch_size)/stride}+1]
    patch_height = int((img.shape[0]-patch_size)/stride) + 1
    # [{(image weight-patch_size)/stride}+1]
    patch_weight = int((img.shape[1]-patch_size)/stride) + 1

    # total patch images = patch_height * patch_weight
    patch_idx = []

    # image column traverse
    for i in range(patch_height+1):
        # get the start and end row index
        s_row = i*stride
        e_row = s_row+patch_size
        
        if e_row > img.shape[0]:
            s_row = img.shape[0] - patch_size
            e_row = img.shape[0]
        
        if e_row <= img.shape[0]:

            # image row traverse
            for j in range(patch_weight+1):
                # get the start and end column index
                start = (j*stride)
                end = start+patch_size
                
                if end > img.shape[1]:
                    start = img.shape[1] - patch_size
                    end = img.shape[1]
                
                if end <= img.shape[1]:
                    tmp = img[s_row:e_row, start:end]  # slicing the image
                    percen = class_percentage_check(
                        tmp)  # find class percentage

                    # take all patch for test images
                    if not patch_class_balance or test == 'test':
                        patch_idx.append([s_row, e_row, start, end])

                    # store patch image indices based on class percentage
                    else:
                        if percen["one_class"] > 19.0:
                            patch_idx.append([s_row, e_row, start, end])
                            
                if end==img.shape[1]:
                    break
            
        if e_row==img.shape[0]:
            break  
            
    return patch_idx


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
    # check for target directory
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)  # making target directory
        except Exception as e:
            print(e)
            raise
    # writing the jason file
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
    def __init__(self, batch_size, channels, ratio=0.3, seed=42, tiles_width=512):
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

        self.tiles_width = tiles_width
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

        # features = [] prev
        rslc0 = []
        rslc1 = []

        labels = []
        labels2 = []

        for i in aug_idx:
            if patch_idx:
                data_p = read_img(data.iloc[i], in_channels = self.channels, patch_idx=patch_idx[i], width=self.tiles_width)
            else:
                data_p = read_img(data.iloc[i], in_channels = self.channels, width=self.tiles_width)

            # masks = np.stack([data_p[1], data_p[2]], axis=-1)     # prev
            # augmented = self.aug(image=data_p[0], mask=masks)
            # features.append(augmented['image'])
            # labels.append(augmented['mask'][:,:,0])
            # labels2.append(augmented['mask'][:,:,1])

            masks = np.stack([data_p[2], data_p[3]], axis=-1)
            augmented = self.aug(image=data_p[0], mask=masks)
            augmented2 = self.aug(image=data_p[1], mask=masks)
            rslc0.append(augmented['image'])
            rslc1.append(augmented2['image'])
            labels.append(augmented['mask'][:,:,0])
            labels2.append(augmented['mask'][:,:,1])
        return rslc0, rslc1, labels, labels2


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
        
        # inputs = []   # prev
        # masks = []
        # masks2 = []

        rslc = []
        rslc2 = []
        masks = []
        masks2 = []

        for i in range(len(batch)):
            if self.patchify:
                data_p = read_img(batch.iloc[i], in_channels = self.in_channels, patch_idx=batch_patch[i], width=self.tile_width)
            else:
                data_p = read_img(batch.iloc[i], in_channels = self.in_channels, width=self.tile_width)
            
            # inputs.append(data_p[0])  # prev
            # masks.append(data_p[1])
            # masks2.append(data_p[2])

            rslc.append(data_p[0])
            rslc2.append(data_p[1])
            masks.append(data_p[2])
            masks2.append(data_p[3])
        
        # augment data using Augment class above if augment is true
        # if self.augment:      prev
        #     if self.patchify:
        #         aug_imgs, aug_masks1, aug_masks2 = self.augment.call(self.data, self.patch_idx) # augment images and mask randomly
        #     else:
        #         aug_imgs, aug_masks1, aug_masks2 = self.augment.call(self.data) # augment images and mask randomly
        #     inputs = inputs + aug_imgs
        #     masks = masks + aug_masks1
        #     masks2 = masks2 + aug_masks2

        if self.augment:
            if self.patchify:
                aug_imgs1, aug_imgs2, aug_masks1, aug_masks2 = self.augment.call(self.data, self.patch_idx) # augment images and mask randomly
            else:
                aug_imgs1, aug_imgs2, aug_masks1, aug_masks2 = self.augment.call(self.data) # augment images and mask randomly
            rslc = rslc + aug_imgs1
            rslc2 = rslc2 + aug_imgs2
            masks = masks + aug_masks1
            masks2 = masks2 + aug_masks2
        
        # inputs, masks, masks2 = self.transform_fn(inputs, masks, masks2, self.num_class)   prev

        rslc, rslc2, masks, masks2 = self.transform_fn(rslc, rslc2, masks, masks2, self.num_class)

        if self.weights != None:

            class_weights = tf.constant(self.weights)
            class_weights = class_weights/tf.reduce_sum(class_weights)
            y_weights = tf.gather(class_weights, indices=tf.cast(masks, tf.int32))#([self.paths[i] for i in indexes])
            y_weights2 = tf.gather(class_weights, indices=tf.cast(masks2, tf.int32))

            # return tf.convert_to_tensor(inputs), [y_weights, y_weights2]  prev
            # return [tf.convert_to_tensor(inputs), tf.convert_to_tensor(inputs)], [y_weights, y_weights2]  prev

            return [tf.convert_to_tensor(rslc), tf.convert_to_tensor(rslc2)], [y_weights, y_weights2]


        # return inputs, [masks, masks2]    prev
        # return [rslc, rslc2], [masks, masks2] prev prev
        return [tf.convert_to_tensor(rslc), tf.convert_to_tensor(rslc2)], [tf.convert_to_tensor(masks), tf.convert_to_tensor(masks2)]
        
    

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
        
        # inputs = []   prev
        # masks = []
        # masks2 = []

        rslc = []
        rslc2 = []
        masks = []
        masks2 = []

        if self.patchify:
            data_p = read_img(self.data.iloc[idx], in_channels = self.in_channels, patch_idx=self.data.patch_idx[idx], width=self.tile_width)
        else:
            data_p = read_img(self.data.iloc[idx], in_channels = self.in_channels, width=self.tile_width)
            
        # inputs.append(data_p[0])  prev
        # masks.append(data_p[1])
        # masks2.append(data_p[2])

        rslc.append(data_p[0])
        rslc2.append(data_p[1])
        masks.append(data_p[2])
        masks2.append(data_p[3])
        
        # inputs, masks, masks2 = self.transform_fn(inputs, masks, masks2, self.num_class)  prev

        rslc, rslc2, masks, masks2 = self.transform_fn(rslc, rslc2, masks, masks2, self.num_class)

        if self.weights != None:

            class_weights = tf.constant(self.weights)
            class_weights = class_weights/tf.reduce_sum(class_weights)
            y_weights = tf.gather(class_weights, indices=tf.cast(masks, tf.int32))#([self.paths[i] for i in indexes])
            y_weights2 = tf.gather(class_weights, indices=tf.cast(masks2, tf.int32))

            # return tf.convert_to_tensor(inputs), y_weights, y_weights2, idx   prev
            return [tf.convert_to_tensor(rslc), tf.convert_to_tensor(rslc2)], [y_weights, y_weights2]

        # return tf.convert_to_tensor(inputs), tf.convert_to_tensor(masks), tf.convert_to_tensor(masks2), idx prev

        return tf.convert_to_tensor(rslc), tf.convert_to_tensor(rslc2), tf.convert_to_tensor(masks), tf.convert_to_tensor(masks2), idx


def get_train_val_dataloader(config):
    """
    Summary:
        read train and valid image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        train and valid dataloader
    """
    # creating csv files for train, test and validation
    if not (os.path.exists(config['train_dir'])):
        data_path_split(config)
    # creating jason files for train, test and validation
    if not (os.path.exists(config["p_train_dir"])) and config['patchify']:
        print("Saving patchify indices for train and test.....")

        # for training
        data = pd.read_csv(config['train_dir'])

        if config["patch_class_balance"]:
            patch_images(data, config, "train_patch_phr_cb_")
        else:
            patch_images(data, config, "train_patch_phr_")

        # for validation
        data = pd.read_csv(config['valid_dir'])

        if config["patch_class_balance"]:
            patch_images(data, config, "valid_patch_phr_cb_")
        else:
            patch_images(data, config, "valid_patch_phr_")
    # initializing train, test and validatinn for patch images
    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config['p_train_dir'], 'r') as j:
            train_dir = json.loads(j.read())
        with open(config['p_valid_dir'], 'r') as j:
            valid_dir = json.loads(j.read())
         
        # selecting which dataset to train and validate   
        train_idx = train_dir['patch_idx']
        valid_idx = valid_dir['patch_idx']
        train_dir = pd.DataFrame.from_dict(train_dir)
        valid_dir = pd.DataFrame.from_dict(valid_dir)
        
    # initializing train, test and validatinn for images
    else:
        print("Loading features and masks directories.....")
        train_dir = pd.read_csv(config['train_dir'])
        valid_dir = pd.read_csv(config['valid_dir'])
        train_idx = None
        valid_idx = None

    print("train Example : {}".format(len(train_dir)))
    print("valid Example : {}".format(len(valid_dir)))

    # create Augment object if augment is true
    if config['augment'] and config['batch_size'] > 1:
        augment_obj = Augment(config['batch_size'], config['in_channels'], tiles_width=config['tiles_size'])
        # new batch size after augment data for train
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch
    else:
        n_batch_size = config['batch_size']
        augment_obj = None

    # get the class weight if weights is true
    if config['weights']:
        weights = tf.constant(config['balance_weights'])
    else:
        weights = None

    # create dataloader object
    train_dataset = MyDataset(train_dir,
                                in_channels=config['in_channels'], patchify=config['patchify'],
                                batch_size=n_batch_size, transform_fn=transform_data, 
                                num_class=config['num_classes'], augment=augment_obj, 
                                weights=weights, patch_idx=train_idx, tile_width=config["tiles_size"])

    val_dataset = MyDataset(valid_dir,
                            in_channels=config['in_channels'], patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=valid_idx, tile_width=config["tiles_size"])
    
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
    if config["evaluation"]:
        var_list = ["eval_dir", "p_eval_dir"]
        patch_name = "eval_patch_phr_cb_"
    else:
        var_list = ["test_dir", "p_test_dir"]
        patch_name = "test_patch_"+ config["experiment"] +"_"
        
    # print(var_list)
        
    if not (os.path.exists(config[var_list[0]])):
        if config["evaluation"]:
            eval_data_path_split(config)
        else:
            data_path_split(config)

    if not (os.path.exists(config[var_list[1]])) and config['patchify']:
        print("Saving patchify indices for test.....")
        data = pd.read_csv(config[var_list[0]])
        patch_images(data, config, patch_name)

    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config[var_list[1]], 'r') as j:
            test_dir = json.loads(j.read()) 
        
        test_idx = test_dir['patch_idx']
        test_dir = pd.DataFrame.from_dict(test_dir)
    else:
        print("Loading features and masks directories.....")
        test_idx = None
        test_dir = pd.read_csv(config[var_list[0]])

    print("test/evaluation Example : {}".format(len(test_dir)))

    test_dataset = MyDataset(test_dir,
                             in_channels=config['in_channels'], patchify=config['patchify'],
                             batch_size=config['batch_size'], transform_fn=transform_data,
                             num_class=config['num_classes'], patch_idx=test_idx, tile_width=config["tiles_size"])

    return test_dataset




if __name__ == '__main__':

    
    print("Complete.")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# from config import get_config
# config = get_config()


# train_dataset, val_dataset = get_train_val_dataloader(config)

    
# check all batch data type
# type_x = []
# type_y = []
# for x,y in train_dataset:
#     tyx = type(x)
#     tyy = type(y)
#     if tyx not in type_x:
#         type_x.append(tyx)
#     if tyy not in type_y:
#         type_y.append(tyy)
# print(type_x)
# print(type_y)
    
    
# check batch length
# print(train_dataset.__len__())
# print(val_dataset.__len__())


# get a specific batch of data
# x, y = train_dataset.__getitem__(1)


# check unique value in valid_dataset
# label_unique = []
# for _, y in val_dataset:
#     val = np.unique(y[0])
#     for i in val:
#         if i not in label_unique:
#             label_unique.append(i)
# print(label_unique)


# check unique value in train_dataset
# label_unique = []
# for _, y in train_dataset:
#     val = np.unique(y[0].numpy())
#     for i in val:
#         if i not in label_unique:
#             label_unique.append(i)
#     break
# print(label_unique)


# check value of each image in a single batch 
# for _, y in train_dataset:
#     val = y[0].numpy()
#     for i in range(len(y[0])):
#         val = np.unique(y[0][i])
#         print(val)
#     break


# read single rslc image
# path = '/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/data/rslc0/features/20131009_5_5.rslc'
# rslc0 = data_reader.readBin(path, width=1024, dataType='floatComplex')

# type(rslc0)

# np.unique(rslc0)

# rslc0_amp = np.abs(rslc0)
# np.unique(rslc0_amp)

# len(np.unique(rslc0_amp))

# scaler = MinMaxScaler((0.0,0.9999999))
# inputs = []
# inputs.append(scaler.fit_transform(rslc0_amp))
# np.unique(inputs)


# plot label

# from utils import display, create_mask
# directory = "/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/plot_val"

# label_unique = []
# for x, y in val_dataset:
#     for i in range(len(x)):
#         display({"RSLC1 AMP": x[0][i],
#                 "RSLC2 AMP ": x[1][i],
#                 "RSLC1 Mask": np.argmax(y[0][i], axis = 2),
#                 "RSLC2 Mask": np.argmax(y[1][i], axis = 2)
#                 }, i, directory, 0, config['experiment'], visualize=True)
#         break
#     break



    
    # check data from Mydataset class
    # train_dir = pd.read_csv("/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/data/train.csv")
    # valid_dir = pd.read_csv("/mnt/hdd2/mdsamiul/waterbody_detection_complex_data/data/valid.csv")

    # train_dataset = MyDataset(train_dir,
    #                             in_channels=3, patchify=False,
    #                             batch_size=10, transform_fn=transform_data, 
    #                             num_class=2, augment=None, 
    #                             weights=None, patch_idx=None, tile_width=512)
    
    #x, y = train_dataset.__getitem__(1)
    # for batch in train_dataset:
    #     x, y = batch
    #     break
    # print(type(x))
    # print(tf.shape(x))
    # print(type(y))
    # print(tf.shape(y))
    # print(train_dataset.__len__())
    
    
    # import matplotlib.pyplot as plt
    # for x,y in train_dataset:
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(x[0][0])
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(y[0][0])
    #     break
    # import matplotlib.pyplot as plt
    # import earthpy.plot as ep
    # import earthpy.spatial as es
    # ax = plt.gca()
    # hillshade = es.hillshade( x, azimuth=180)
    # ep.plot_bands(
    #     x,
    #     cbar=False,
    #     cmap="terrain",
    #     title=title[i],
    #     ax=ax
    #     )
    
    # for x,y in train_dataset:
    #     print(y[0][1].shape)
    #     ab = np.argmax([y[0][1]], axis=3)[0]
    #     print(ab.shape)
    #     break

# train_dataset = MyDataset(train_dir,
#                             in_channels=3, patchify=False,
#                             batch_size=10, transform_fn=transform_data, 
#                             num_class=2, augment=None, 
#                             weights=None, patch_idx=None, tile_width=512)


# for batch in train_dataset:
#     x, y = batch
#     break
# print(type(x))
# print(type(y))
# print(train_dataset.__len__())

# print(x.shape)

# len(y)

# type(y[0])
# print(y[0].shape)


# for patchifying 

# from config import get_config
# config = get_config()
# with open(config['p_train_dir'], 'r') as j:
#     train_dir = json.loads(j.read())
# with open(config['p_valid_dir'], 'r') as j:
#     valid_dir = json.loads(j.read())
         
# train_idx = train_dir['patch_idx']
# valid_idx = valid_dir['patch_idx']
# train_dir = pd.DataFrame.from_dict(train_dir)
# valid_dir = pd.DataFrame.from_dict(valid_dir)



# print(label_unique)