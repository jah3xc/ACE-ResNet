import argparse
import logging

def main():
    args = init()

def init():
    levelMap = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "CRITICAL": logging.CRITICAL,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("ace_examples_dir")
    parser.add_argument("-l", "--log", choices=levelMap.keys(), default="ERROR")

    args = vars(parser.parse_args())

    logging.basicConfig(level=args["log"])
    del args["log"]

    return args


if __name__ == "__main__":
    main()