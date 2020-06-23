import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import time
import pprint
import numpy as np
import random
import argparse
import torch.backends.cudnn as cudnn
import utils
import logging
from PIL import Image
from mmcv import Config
from models.basic_operations import operation_canditates
from attack import *
import models


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./experiments/RobNet_cifar10_experiments/config_search.py',
                    help='location of the config file')
parser.set_defaults(augment=True)
args = parser.parse_args()


def cifar10(data_root, batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                        (4,4,4,4), mode='constant', value=0).squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader

def sampleGenotypeList():
    genotype_list_str = []
    genotype_list = []
    # If we want to avoid specific operations, random search in given list__choice_list
    # choice_list = ['01', '10', '11']
    for i in range(14):
        curr_operation = random.choice(list(operation_canditates.keys()))
#         curr_operation = random.choice(choice_list)
        curr_operation_idx = int(curr_operation, 2)
        genotype_list.append(curr_operation_idx)
        genotype_list_str.append(curr_operation)
    return genotype_list, genotype_list_str


def train(net, trainloader, optimizer, genotype_list, adv=True):

    losses = utils.AverageMeter(0)
    top1 = utils.AverageMeter(0)
    top5 = utils.AverageMeter(0)

    logger = logging.getLogger('global_logger')
    trainloader_iterator = iter(trainloader)
    try:
        inputs, targets = next(trainloader_iterator)
    except StopIteration:
        trainloader_iterator = iter(dataloader)
        inputs, targets = next(trainloader_iterator)
    inputs, targets = inputs.cuda(), targets.cuda()


    if not adv:
        outputs = net(inputs)
    else:
        outputs, inputs_adv = net(inputs, targets, genotype_list)
    loss = F.cross_entropy(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, 5))
    num = inputs.size(0)
    losses.update(loss.item(), num)
    top1.update(prec1.item(), num)
    top5.update(prec5.item(), num)

    logger.info("Genotypes: {}".format(genotype_list))
    logger.info('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
    .format(loss=losses, top1=top1, top5=top5))


def search():
    global cfg

    cfg = Config.fromfile(args.config)

    cfg.save = '{}/{}-{}-{}'.format(cfg.save_path, cfg.model, cfg.dataset,
                                    time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(cfg.save)
    logger = utils.create_logger('global_logger', cfg.save + '/log.txt')

    # Set cuda device & seed
    torch.cuda.set_device(cfg.gpu)
    np.random.seed(cfg.seed)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(cfg.seed)

    I = 20000
    trainloader = cifar10(cfg.dataset_param.data_root, cfg.dataset_param.batch_size, cfg.dataset_param.num_workers)
    net = models.model_search_entry(cfg)
    net = net.cuda()
    net_adv = AttackPGD(net, cfg.attack_param)
    optimizer = torch.optim.SGD(net_adv.parameters(), lr=0.1, momentum=0.9)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    net_adv.train()

    cfg.netpara = sum(p.numel() for p in net_adv.parameters()) / 1e6
    logger.info('Config: {}'.format(pprint.pformat(cfg)))

    for k in range(I):
        genotype_list, genotype_list_str = sampleGenotypeList()
        train_scheduler.step()
        train(net_adv, trainloader, optimizer, genotype_list, adv=True)

    PATH = './my_experiments/supernet_20k.pth'
    torch.save(net_adv.state_dict(), PATH)


if __name__ == "__main__":

    search()

