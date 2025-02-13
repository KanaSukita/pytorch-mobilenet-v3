import torch.backends.cudnn as cudnn

from dataset.tinyimagenet import TinyImagenet
from mobilenetv3 import MobileNetV3
from train import Train
from utils import parse_args, create_experiment_dirs


def main():
    # Parse the JSON arguments
    config_args = parse_args()

    # Create the experiment directories
    #_, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(
    #    config_args.experiment_dir)

    model = MobileNetV3(n_class=200, input_size=64, classify=config_args.classify)

    if config_args.cuda:
        model.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    data = TinyImagenet(config_args)
    print("Data loaded successfully\n")

    trainer = Train(model, data.trainloader, data.testloader, config_args)

    if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            pass

    if config_args.to_test:
        print("Testing...")
        trainer.test(data.testloader)
        print("Testing Finished\n")


if __name__ == "__main__":
    main()