import argparse
import logging
from util import load_mat
from patches import extract_patches
from pathlib import Path
from cnn import train_model
from ACE import ace_transform_samples
import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = 1

def main():
    args = init()
    logger = logging.getLogger(__name__)

    ###########
    # Load the Data
    ###########
    data_path, gt_path = args["dataset"], args["groundTruth"]
    if not Path(data_path).absolute().is_file() or not Path(gt_path).absolute().is_file():
        logger.critical("Dataset or GT do not exist!")
    data, ground_truth = load_mat(data_path, gt_path)
    logger.info("Loaded data...")

    ############
    # Extract Patches
    ############
    window_size = args["windowSize"]
    stride = args["stride"]
    maxPatches = args["maxPatches"] if "maxPatches" in args else None
    samples, labels = extract_patches(data, ground_truth, window_size, stride, maxPatches=maxPatches)
    logger.info("Extracted {} patches".format(len(samples)))

    ###########
    # ACE If necessary
    ###########
    ACE = args["ace"]
    cpu_count = args["cpu_count"] if "cpu_count" in args else os.cpu_count()
    if ACE:
        samples, labels = ace_transform_samples(samples, labels, data, ground_truth, cpu=cpu_count)
    

    ###########
    # Train the Network
    ###########
    trainParams = args["training"] if "training" in args else {}
    buildParams = args["building"] if "building" in args else{}
    model = train_model(samples, labels, window_size, buildParams, trainParams)


def init():
    levelMap = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "CRITICAL": logging.CRITICAL,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("configFile", type=str)
    parser.add_argument("-l", "--log", choices=levelMap.keys(), default="ERROR")

    args = vars(parser.parse_args())

    logging.basicConfig(level=args["log"])
    del args["log"]

    filename = args["configFile"]
    params = json.load(open(filename))

    return params


if __name__ == "__main__":
    main()