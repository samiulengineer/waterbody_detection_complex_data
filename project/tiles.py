import os
import yaml
import glob
import pathlib
import numpy as np
import pandas as pd
from mrc_insar_common.data import data_reader

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

def save_tiles(path, patch_size=512, stride=512, width=11361, height=10000, out_path=None, label=False):
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
    patch_height = int((slc.shape[0]-patch_size)/stride)+1 # [{(image height-patch_size)/stride}+1]
    patch_width = int((slc.shape[1]-patch_size)/stride)+1 # [{(image width-patch_size)/stride}+1]
    
    # image column traverse
    for i in range(patch_height):
        s_row = i*stride
        e_row = s_row+patch_size
        for j in range(patch_width):
            start = (j*stride)
            end = start+patch_size
            tmp = slc[s_row:e_row, start:end]
            if label:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(j)+".ras"
                writeRas(tmp, f_name, cmap)
            else:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(j)+".rslc"
                data_reader.writeBin(f_name, tmp, "floatComplex")
    
    # if shape does not match with patch multiplication
    if (patch_height*patch_size)<slc.shape[0]:
        s_row = slc.shape[0]-patch_size
        e_row = slc.shape[0]
        for j in range(patch_width):
            start = (j*stride)
            end = start+patch_size
            tmp = slc[s_row:e_row, start:end]
            if label:
                f_name = out_path+slc_id+"_"+str(patch_height)+"_"+str(j)+".ras"
                writeRas(tmp, f_name, cmap)
            else:
                f_name = out_path+slc_id+"_"+str(patch_height)+"_"+str(j)+".rslc"
                data_reader.writeBin(f_name, tmp, "floatComplex")
    
    if (patch_width*patch_size)<slc.shape[1]:
        for i in range(patch_height):
            s_row = i*stride
            e_row = s_row+patch_size
            start = slc.shape[1]-patch_size
            end = slc.shape[1]
            tmp = slc[s_row:e_row, start:end]
            if label:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(patch_width)+".ras"
                writeRas(tmp, f_name, cmap)
            else:
                f_name = out_path+slc_id+"_"+str(i)+"_"+str(patch_width)+".rslc"
                data_reader.writeBin(f_name, tmp, "floatComplex")

if __name__ == '__main__':

    wid, hgt = 11361, 10820  # from the masterpar.xml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    rslcs = sorted(glob.glob(config["dataset_dir"]+"rslc/*.rslc.notopo"))
    label = sorted(glob.glob(config["dataset_dir"]+"label/*.ras"))

    data_paths = {}

    assert len(rslcs)== len(label), "Invalid number of label or rslcs" # must have same number of rslc and label

    for i in range(len(rslcs)-1):
        out_path = config["dataset_dir"]+"rslc"+str(i)+"/"
        save_tiles(rslcs[i], patch_size=512, stride=512, width=wid, height=hgt, out_path=(out_path+"features/"))
        save_tiles(label[i], patch_size=512, stride=512, width=wid, height=hgt, out_path=(out_path+"label/"), label=True)
        data_paths["rslc"+str(i)] = sorted(glob.glob((out_path+"features/*.rslc")))
        data_paths["rslc"+str(i)+"_label"] = sorted(glob.glob((out_path+"label/*.ras")))
    
    df = pd.DataFrame(data_paths)
    df.to_csv((config["dataset_dir"]+"tiles_path.csv"), index=False)
