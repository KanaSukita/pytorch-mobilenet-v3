import torch.utils.data as data
import os
from PIL import Image
import numpy as np

class CustomDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        
        if train:
            classes, class_to_idx = self.find_classes(os.path.join(root, 'train'))
            samples = self.make_dataset_multi_folders(os.path.join(root, 'train'), class_to_idx)
        
        else:
            #classes, class_to_idx = self.find_classes(os.path.join(root, 'val'))
            #samples = self.make_dataset(os.path.join(root, 'val'), class_to_idx)
            classes, class_to_idx = self.find_classes(os.path.join(root, 'train'))
            samples = self.make_dataset_single_folder(os.path.join(root, 'val'), class_to_idx)


        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def has_file_allowed_extension(self, filename):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in self.extensions)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset_multi_folders(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.has_file_allowed_extension(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    def make_dataset_single_folder(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        files = os.listdir(dir)
        for file in files:
            if file.endswith('txt'):
                fh = open(os.path.join(dir, file), 'r')
                for line in fh:  
                    words = line.split() 
                    path = os.path.join(dir, 'images')
                    path = os.path.join(path, words[0])
                    item = (path, class_to_idx[words[1]])
                    images.append(item)

        return images