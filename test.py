from utils import calc_dataset_stats
from dataset import custom

from dataset.tinyimagenet import TinyImagenet
from model import MobileNetV3
from train import Train
from utils import parse_args, create_experiment_dirs


def main():

    model = mobilenetv3()

    print("Loading Data...")
    data = TinyImagenet(config_args)
    print("Data loaded successfully\n")

    trainer = Train(model, data.trainloader, data.testloader, config_args)