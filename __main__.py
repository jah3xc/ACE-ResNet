import argparse
import logging
from util import load_mat

def main():
    args = init()
    logger = logging.getLogger(__name__)

    data, ground_truth = load_mat(args["data_path"], args["gt_path"])
    logger.info("Loaded data...")

def init():
    levelMap = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "CRITICAL": logging.CRITICAL,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("gt_path")
    parser.add_argument("-l", "--log", choices=levelMap.keys(), default="ERROR")

    args = vars(parser.parse_args())

    logging.basicConfig(level=args["log"])
    del args["log"]

    return args


if __name__ == "__main__":
    main()