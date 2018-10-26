import numpy as np
from util import calc_cov, calc_mean
from tqdm import tqdm



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