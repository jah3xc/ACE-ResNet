import numpy as np
from util import calc_cov, calc_mean
from patches import normalize_patch as normalize
from tqdm import tqdm
from multiprocessing import Pool
import os
import logging

def get_mean_signatures(data, ground_truth):
    unique, counts = np.unique(ground_truth, return_counts=True)
    Xdim, Ydim, bands = data.shape
    num_classes = len(unique)
    mean_signatures = np.empty([num_classes, bands])
    for class_num, count in zip(unique, counts):
        # get all pixels in this class
        mask = np.where(ground_truth == class_num, 1, 0)
        examples = np.empty([count, bands])
        x = 0
        # multiply the mask with the data
        for i in range(Xdim):
            for j in range(Ydim):
                if mask[i, j] == 1:
                    examples[x] = data[i, j]
                    x += 1
        # get the mean sig
        mean_sig = np.sum(examples, axis=0) / count
        # save it
        mean_signatures[class_num] = normalize(mean_sig)
    return mean_signatures

def ace_transform_samples(samples, labels, data, ground_truth, cpu = os.cpu_count()):
    logger = logging.getLogger(__name__)
    # get the mean sig
    mean_signatures = get_mean_signatures(data, ground_truth)
    # run each samples through all mean signatures
    _, Xdim, Ydim, _ = samples.shape
    
    num_classes = len(np.unique(ground_truth))
    
    ace_samples = np.empty([1, Xdim, Ydim, num_classes])
    task_list = list(ace_generator(samples, labels, mean_signatures))
    size = len(task_list) // os.cpu_count()
    labels = []
    logger.info("Spawning Tasks")
    with Pool(cpu) as pool:
        logger.info("All tasks spawned!")
        for result in tqdm(pool.imap_unordered(transform_sample, task_list, chunksize=size), total=len(task_list), desc="Running ACE"):
            ace_sample, label = result
            ace_sample = np.array([ace_sample])
            ace_samples = np.concatenate((ace_samples, ace_sample), axis=0)
            labels.append(label)
        pool.close()
        pool.join()
    logger.info("Closed pool")
    ace_samples = ace_samples[1:]
    labels = np.array(labels)
    return ace_samples, labels

def ace_generator(samples, labels, mean_signatures):
    for i,j in zip(samples, labels):
        yield i, j, mean_signatures

def transform_sample(params):
    logger = logging.getLogger(__name__)
    sample, label, mean_signatures = params
    Xdim, Ydim, _ = sample.shape
    num_classes = len(mean_signatures)
    ace_samp = np.empty([Xdim, Ydim, num_classes])
    for j, mean_sig in enumerate(mean_signatures):
        ace_samp[:,:, j] = ACE(sample, mean_sig)
        logger.debug("Finished for class {}".format(j))
    logger.debug("Finished a sample!")
    return ace_samp, label



def ACE(data, s, window_size = 3):
    """
    ACE Global Algorithm
    :param data: the data to run on
    :param s: the signal to match
    :return: the results of ace local
    """
    step = window_size // 2
   
    Xdim, Ydim, Bands = data.shape
    # calc valid values
    min_i = min_j = step
    max_i, max_j = Xdim - step - 1, Ydim - step - 1

    ace = np.zeros((Xdim,Ydim))
    for i, row in enumerate(data):
        if i not in range(min_i, max_i):
            continue
        for j, x in enumerate(row):
            if j not in range(min_j, max_j):
                continue
            # calc window
            window = data[(i-step):(i+step+1), (j-step):(j+step+1), :]
            # calc mu and sigma
            mu = calc_mean(window)
            sigma = calc_cov(window)
            
            
            # what if its not invertable ... ?
            if( np.linalg.det( sigma ) < 0.00001 ):
                Pad = 0.1 * np.identity(Bands); # make an "identity matrix" and put some small padding along it!
                sigma = np.add( sigma , Pad );
                
            sigma_inv = np.linalg.inv(sigma)

            # calc num
            numerator = np.square(np.matmul(np.matmul(s-mu,sigma_inv), np.matrix.transpose(x-mu)))
            # calc denom
            lowerl = np.matmul( np.matmul(s-mu,sigma_inv),np.matrix.transpose(s-mu))
            lowerr = np.matmul( np.matmul(x-mu,sigma_inv), np.matrix.transpose(x-mu) )
            denominator = lowerl*lowerr
            # divide and store!
            ace[i,j] = numerator / denominator
    return ace