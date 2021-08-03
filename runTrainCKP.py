import os
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms, utils

from torchlib.datasets.fersynthetic import SyntheticFaceDataset, SecuencialSyntheticFaceDataset
from torchlib.datasets.factory import FactoryDataset
from torchlib.attentionnet import AttentionNeuralNet, AttentionSTNNeuralNet, AttentionGMMNeuralNet, \
    AttentionGMMSTNNeuralNet

from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

import datetime
from argparse import ArgumentParser
from aug import get_transforms_aug, get_transforms_det
from forStudy.SyntheticFaceDataset import *

# parameters
DATABACK = None  # ~/.datasets/coco'
DATA = '~/.datasets'
NAMEDATASET = 'ckp_by_myself'
PROJECT = '../out/attnet'
EPOCHS = 5
TRAINITERATION = 288000
TESTITERATION = 2880
BATCHSIZE = 32  # 32, 64, 128, 160, 200, 240
LEARNING_RATE = 0.0001
MOMENTUM = 0.5
PRINT_FREQ = 100
WORKERS = 4
RESUME = 'model_best.pth.tar'  # chk000000, model_best
GPU = 0
NAMEMETHOD = 'attnet'  # attnet, attstnnet, attgmmnet, attgmmstnnet
ARCH = 'ferattention'  # ferattention, ferattentiongmm, ferattentionstn
LOSS = 'attloss'
OPT = 'adam'
SCHEDULER = 'fixed'
NUMCLASS = 8  # 6, 7, 8
NUMCHANNELS = 3
DIM = 32
SNAPSHOT = 10
IMAGESIZE = 64
KFOLD = 0
NACTOR = 10
BACKBONE = 'preactresnet'  # preactresnet, resnet, cvgg

EXP_NAME = 'feratt_' + NAMEMETHOD + '_' + ARCH + '_' + LOSS + '_' + OPT + '_' + NAMEDATASET + '_dim' + str(DIM) + '_bb' + BACKBONE + '_fold' + str(KFOLD) + '_000'
# experiment name


def main():
    print('开始训练')
    # parameters
    imsize = IMAGESIZE
    parallel = False
    num_classes = NUMCLASS
    num_channels = NUMCHANNELS
    dim = DIM
    view_freq = 1
    trainiteration = TRAINITERATION  # not use this currently
    testiteration = TESTITERATION  # not use this currently
    no_cuda = False
    seed = 1
    finetuning = False
    balance = False
    fname = NAMEMETHOD
    fnet = {
        'attnet': AttentionNeuralNet,
        'attstnnet': AttentionSTNNeuralNet,
        'attgmmnet': AttentionGMMNeuralNet,
        'attgmmstnnet': AttentionGMMSTNNeuralNet
    }

    network = fnet[fname](
        patchproject=PROJECT,
        nameproject=EXP_NAME,
        no_cuda=no_cuda,
        parallel=parallel,
        seed=seed,
        print_freq=PRINT_FREQ,
        gpu=GPU,
        view_freq=view_freq,
    )

    network.create(
        arch=ARCH,
        num_output_channels=DIM,
        num_input_channels=NUMCHANNELS,
        loss=LOSS,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        optimizer=OPT,
        lrsch=SCHEDULER,
        pretrained=finetuning,
        size_input=imsize,
        num_classes=num_classes
    )

    # resume
    network.resume(os.path.join(network.pathmodels, RESUME))
    cudnn.benchmark = True

    kfold = KFOLD
    nactores = NACTOR
    idenselect = np.arange(nactores) + kfold * nactores

    # datasets
    # training dataset
    # SyntheticFaceDataset, SecuencialSyntheticFaceDataset
    train_data = CKPSyntheticFaceDataset(
        data=FactoryDataset.factory(
            pathname=DATA,
            name=NAMEDATASET,
            subset=FactoryDataset.training,
            idenselect=idenselect,
            download=False
        ),
        # pathnameback=args.databack,
        ext='jpg',
        count=trainiteration,
        num_channels=NUMCHANNELS,
        iluminate=True, angle=30, translation=0.2, warp=0.1, factor=0.2,
        transform_data=get_transforms_aug(imsize),
        transform_image=get_transforms_det(imsize),
    )

    num_train = len(train_data)
    if balance:
        labels, counts = np.unique(train_data.labels, return_counts=True)
        weights = 1 / (counts / counts.sum())
        samples_weights = np.array([weights[x] for x in train_data.labels])
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    else:
        sampler = SubsetRandomSampler(np.random.permutation(num_train))

    train_loader = DataLoader(
        train_data,
        batch_size=BATCHSIZE,
        num_workers=WORKERS,
        pin_memory=network.cuda,
        drop_last=True,
        sampler=sampler,
        # shuffle=True
    )

    # validate dataset
    # SyntheticFaceDataset, SecuencialSyntheticFaceDataset
    val_data = CKPSyntheticFaceDataset(
        data=FactoryDataset.factory(
            pathname=DATA,
            name=NAMEDATASET,
            idenselect=idenselect,
            subset=FactoryDataset.validation,
            download=False
        ),
        # pathnameback=args.databack,
        ext='jpg',
        count=testiteration,
        num_channels=NUMCHANNELS,
        iluminate=True, angle=30, translation=0.2, warp=0.1, factor=0.2,
        transform_data=get_transforms_aug(imsize),
        transform_image=get_transforms_det(imsize),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=network.cuda,
        drop_last=False
    )

    # print neural net class
    print('SEG-Torch: {}'.format(datetime.datetime.now()))
    print(network)

    # training neural net
    network.fit(train_loader, val_loader, EPOCHS, SNAPSHOT)

    print("Optimization Finished!")
    print("DONE!!!")


if __name__ == '__main__':
    main()
