import argparse

DATA_DIRECTORY = '../datasets'
DATASET = 'VOC2007'
SET = 'train'
INPUT_SIZE = 224
MODE = 'train'

parser = argparse.ArgumentParser(description="DeepLabV3")
parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the PASCAL VOC dataset.")
parser.add_argument("--dataset", type=str, default=DATASET,
                    help="Name of the dataset.")
parser.add_argument("--set", type=str, default=SET,
                    help="Name of the dataset.")
parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                    help="height and width of images.")
parser.add_argument("--mode", type=str, default=MODE,
                    help="train or test.")
args = parser.parse_args()
