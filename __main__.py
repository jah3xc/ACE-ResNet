import argparse
import logging
from util import load_mat, extract_patches
from pathlib import Path
from cnn import train_model

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
    samples, labels = extract_patches(data, ground_truth, window_size)


    ###########
    # Train the Network
    ###########
    model = train_model(samples, labels)


def init():
    levelMap = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "CRITICAL": logging.CRITICAL,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group()
    required.add_argument("-d","--dataset", required = True)
    required.add_argument("-g", "--groundTruth", required = True)
    optional = parser.add_argument_group()
    optional.add_argument("--ace", action="store_true")
    optional.add_argument("-w", "--windowSize", default=11)
    optional.add_argument("-l", "--log", choices=levelMap.keys(), default="ERROR")

    args = vars(parser.parse_args())

    logging.basicConfig(level=args["log"])
    del args["log"]

    return args


if __name__ == "__main__":
    main()