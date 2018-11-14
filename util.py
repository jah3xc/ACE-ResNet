import scipy.io
import numpy as np
from copy import deepcopy
from PIL import Image
import keras
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import logging
import sys


def extract_patches(data, ground_truth, window_size, maxPatches = None):
    logger = logging.getLogger(__name__)
    Xdim, Ydim, bands = data.shape
    if maxPatches is None:
        num_patches = Xdim * Ydim
    else:
        num_patches = maxPatches
    # init patches and labels
    patches = np.empty([1, window_size, window_size, bands])
    labels = []
    # init progress bar
    progress = 0
    #########
    # Function to add a patch
    # to the list of patches
    ########
    
    def add_patch(result):
        nonlocal patches
        nonlocal labels
        nonlocal progress
        nonlocal num_patches
        progress += 1
        sys.stdout.write("{}/{}  {:.2f}%".format(progress, num_patches, progress / num_patches))
        sys.stdout.flush()
        patch, label = result
        patch = np.array([patch])
        patches = np.concatenate((patches, patch), axis=0)
        labels.append(label)

    ########
    # Spawn Pool
    ########
    num_patches_processed = 0
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=num_patches, desc="Spawning tasks") as pbar:
            for x in range(Xdim):
                for y in range(Ydim):
                    # check for maxPatches
                    if num_patches_processed >= num_patches:
                        logger.debug("Reached maxPatches")
                        break
                    # give to workers
                    pool.apply_async(
                        extract_patch,
                        (data, x, y, window_size, ground_truth[x,y]),
                        callback=add_patch
                    )
                    num_patches_processed += 1
                    pbar.update(1)
                # check for maxPatches
                if num_patches_processed >= num_patches:
                    break
        pool.close()
        pool.join()
    # throw away the first patch
    patches = patches[1:]
    # convert to numpy categorical
    labels = np.array(labels)
    labels = keras.utils.to_categorical(labels)
    logger.debug("Shape of Labels: {}".format(labels.shape))
    logger.info("Number of patches found: {}".format(len(patches)))
    return patches, labels

def extract_patch(data, x, y, window_size, label):
    return np.zeros([window_size, window_size, data.shape[2]]), label


def load_mat(data_path, gt_path):
    
    mat = scipy.io.loadmat(data_path)
    data = mat['indian_pines']
    
    
    mat2 = scipy.io.loadmat(gt_path)
    data2 = mat2['indian_pines_gt']
    
    return data, data2

def calc_mean(data):
    """
    Calculate and return the mean of the data
    :param data: the data
    """
    Xdim, Ydim, Bands = data.shape
    gm = np.zeros(shape=(1,Bands)) # allocate our data
    for i in range(Xdim): 
        for j in range(Ydim):
            gm = gm + data[i,j,:]
    gm = gm / (Xdim * Ydim)
    return gm

def calc_cov(data):
    """
    Calculate and return the 
    covariance matrix
    :param data: the data
    """
    gm = calc_mean(data)
    Xdim, Ydim, Bands = data.shape
    C = np.zeros(shape=(Bands,Bands))
    for i in range(Xdim): 
        for j in range(Ydim):
            DiffVec = data[i,j,:] - gm
            C = C + np.outer( DiffVec , DiffVec )
    sigma = C = C / (Xdim * Ydim - 1)
    return sigma

def show_ace(ace, filename="ace.png"):  
    """
    Show results of ACE
    :param ace: results of ace
    """
    # show to screen 
    show = input("Enter 0 for raw or 1 for scaled output: ")
    if int(show) not in (0,1):
        print("Invalid Selection!")

    img_data = deepcopy(ace)

    if int(show) == 1:
        img_data = img_data / np.max(img_data)
    img_data *= 255
    img_data = img_data.astype(np.uint8)
    Image.fromarray(img_data).resize((200,200)).save(filename)