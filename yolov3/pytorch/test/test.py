import sys
sys.path.append("..")
from utils.parse_config import *


def test_parse_model_config(path):
    """test code for this function
    """
    networkStructure = parse_model_config(path)
    print(len(networkStructure), networkStructure[0], networkStructure[1],
          networkStructure[2], networkStructure[3])
    return


if __name__ == "__main__":
    configFile = "../config/yolov3.cfg"
    test_parse_model_config(configFile)