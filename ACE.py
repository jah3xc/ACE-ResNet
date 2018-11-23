import numpy as np
from util import calc_cov, calc_mean
from patches import normalize_patch as normalize
from tqdm import tqdm

def get_mean_signatures(data, ground_truth):
    unique, counts = np.unique(ground_truth, return_counts=True)
    _,_,bands = data.shape
    num_classes = len(unique)
    mean_signatures = np.empty([num_classes, bands])
    for i, class_num, count in enumerate(zip(unique, counts)):
        # get all pixels in this class
        class_mask = np.where(ground_truth == i, 1, 0)
        # multiply the mask with the data
        masked_data = np.multiply(data, class_mask)
        # get the mean sig
        mean_sig = np.sum(masked_data, axis=0) / np.array([count] * len(masked_data))
        # save it
        mean_signatures[class_num] = normalize(mean_sig)
    return mean_signatures

def ace_transform_samples(samples, labels, data, ground_truth):
    # get the mean sig
    mean_signatures = get_mean_signatures(data, ground_truth)
    # run each samples through all mean signatures
    num_samples, Xdim, Ydim, bands = samples.shape
    num_classes = len(np.unique(ground_truth))
    ace_samples = np.empty([num_samples, Xdim, Ydim, num_classes])

    with tqdm(total=len(samples), desc="Running ACE") as pbar:
        for i, samp in enumerate(samples):
            ace_samp = np.empty([Xdim, Ydim, num_classes])
            for j, mean_sig in enumerate(mean_signatures):
                ace_samp[:,:, j] = ACE(samp, mean_sig)
            ace_samples[i] = ace_samp
            pbar.update(1)

    return ace_samples


def ACE(data, s):
    """
    ACE Global Algorithm
    :param data: the data to run on
    :param s: the signal to match
    :return: the results of ace local
    """
    window_size = int(input("Enter input window size: "))
    step = window_size // 2
    print("Using window size of {}".format(window_size))
    Xdim, Ydim, Bands = data.shape
    # calc valid values
    min_i = min_j = step
    max_i, max_j = Xdim - step - 1, Ydim - step - 1

    ace = np.zeros((Xdim,Ydim))
    for i, row in tqdm(enumerate(data)):
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