from utils import calc_dataset_stats
from dataset import custom

from dataset.tinyimagenet import TinyImagenet
from mobilenetv3 import MobileNetV3
from train import Train
from utils import parse_args, create_experiment_dirs
import torch
import os

if __name__ == '__main__':

    model = MobileNetV3(n_class=200, input_size=64)
    model.cuda()
    model_dict = model.state_dict()
    config_args = parse_args()

    print("Loading Data...")
    data = TinyImagenet(config_args)
    print("Data loaded successfully\n")

    trainer = Train(model, data.trainloader, data.testloader, config_args)
    trainer.train()

