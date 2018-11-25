import scipy.io
import numpy as np
from copy import deepcopy
from PIL import Image
# import keras
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import logging
import sys


def patch_generator(data, Xdim, Ydim, window_size, stride, ground_truth):
    # calc valid values
    step = window_size // 2
    min_i = min_j = step
    max_i, max_j = Xdim - step - 1, Ydim - step - 1

    x_range = np.arange(min_i, max_i, stride)
    y_range = np.arange(min_j, max_j, stride)
    np.random.shuffle(x_range)
    np.random.shuffle(y_range)
    for x in x_range:
        for y in y_range:
            yield data, x, y, window_size, ground_truth[x, y]
                
def extract_patches(data, ground_truth, window_size, stride, maxPatches = None):
    logger = logging.getLogger(__name__)
    Xdim, Ydim, bands = data.shape
    if maxPatches is None:
        num_patches = Xdim * Ydim
    else:
        num_patches = maxPatches
    # init patches and labels
    patches = np.empty([1, window_size, window_size, bands])
    labels = []

    ########
    # Spawn Pool
    ########
    iterable = list(patch_generator(data, Xdim, Ydim, window_size, stride, ground_truth))
    iterable = iterable[:num_patches]
    chunk_size = len(iterable) // os.cpu_count()
    with tqdm(total=len(iterable), desc="Extracting patches") as pbar:
        with Pool(os.cpu_count()) as pool:
            for result in pool.imap_unordered(extract_patch, iterable, chunksize=chunk_size):
                pbar.update(1)
                patch, label = result
                patch = np.array([patch])
                patch = normalize_patch(patch)
                patches = np.concatenate((patches, patch), axis=0)
                labels.append(label)
            pool.close()
            pool.join()
    # throw away the first patch
    patches = patches[1:]
    # convert to numpy categorical
    labels = np.array(labels)
    # labels = keras.utils.to_categorical(labels)
    logger.debug("Shape of Labels: {}".format(labels.shape))
    logger.info("Number of patches found: {}".format(len(patches)))

    return patches, labels

def normalize_patch(x, range = 1):
        """
        Standardize the sample in the range 
        :param x: the sample
        :param range: the range within to normalize
        :return: normalized sample
        """
        img_f = x.astype('float')
        max_val = np.max(x)
        min_val = np.min(x)
        img_ac = (img_f - min_val) / (max_val - min_val) * 2. * range - range
        return img_ac

def extract_patch(params):
    data, x, y, window_size, label = params
    Xdim, Ydim, numBands = data.shape
    offset = window_size // 2
    window = data[(x - offset):(x + offset + 1), (y - offset):(y + offset + 1),:]
    return window, label