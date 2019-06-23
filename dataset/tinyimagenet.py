import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import calc_dataset_stats
from dataset.custom import CustomDataset

# Example DataLoader on CIFAR-10
class TinyImagenet:
    def __init__(self, args):
        #mean, std = calc_dataset_stats(CustomDataset(root='/Users/wenjunzhang/Desktop/Workspace/dataset/tiny-imagenet-200', 
        #                                             train=True))

        train_transform = transforms.Compose(
            [transforms.RandomCrop(args.img_height),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.3, 0.3, 0.3),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.trainloader = DataLoader(CustomDataset(root='/home/wenjun/Workspace/dataset/tiny-imagenet-200', train=True,
                                                                   transform=train_transform),
                                      batch_size=args.batch_size,
                                      shuffle=args.shuffle, num_workers=args.dataloader_workers,
                                      pin_memory=args.pin_memory)

        self.testloader = DataLoader(CustomDataset(root='/home/wenjun/Workspace/dataset/tiny-imagenet-200', train=False,
                                                                  transform=test_transform),
                                      batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.dataloader_workers,
                                      pin_memory=args.pin_memory)

    def plot_random_sample(self):
        # Get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()
        print(images[0])
        exit(1)
        # Show images
        grid = torchvision.utils.make_grid(images)
        img = grid / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        # Print labels
        print(' '.join('%5s' % CIFAR100_LABELS_LIST[labels[j]] for j in range(len(labels))))
        
CIFAR10_LABELS_LIST = [
'airplane',
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck'
]