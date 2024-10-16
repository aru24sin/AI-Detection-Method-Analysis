import os
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets

from .datasets import dataset_folder


def get_dataset(opt):
    ## 7/15/24, modify this to make it suitable for GenImage CNN detector
    ## Since there are only two classes: ai and nature, so the root path
    ## should be "dataset/train" or "dataset/val"

    print(f"Class path: {opt.dataroot}")

    if not os.path.exists(opt.dataroot):
        raise FileNotFoundError(f"data root directory not found: {opt.dataroot}")

    dset = dataset_folder(opt, opt.dataroot)

    return dset
    
    
    #dset_lst = []
    #for cls in opt.classes:
    #    class_path = os.path.join(opt.dataroot, cls)
    #    print(f"Class path: {class_path}")

    #   # Ensure the class directory exists
    #   if not os.path.exists(class_path):
    #        raise FileNotFoundError(f"Class directory not found: {class_path}")

    #    dset = dataset_folder(opt, class_path)
    #    dset_lst.append(dset)

    #return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
