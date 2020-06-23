import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import random
import argparse
import time
import pprint
import torch.backends.cudnn as cudnn
import utils
from PIL import Image
from mmcv import Config
from models.basic_operations import operation_canditates
from attack import *
import models

Debug = False
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./experiments/RobNet_cifar10_experiments/config_finetune.py',
                    help='location of the config file')
parser.add_argument('--model_name', type=str, default='default_model.pth', help="name of stored model as pth file")
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
        # curr_operation = random.choice(choice_list)
        curr_operation_idx = int(curr_operation, 2)
        genotype_list.append(curr_operation_idx)
        genotype_list_str.append(curr_operation)
    return genotype_list, genotype_list_str

def finetune():
    global cfg

    cfg = Config.fromfile(args.config)
    cfg.save = '{}/{}-{}-{}'.format(cfg.save_path, cfg.model, cfg.dataset,
                                    time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(cfg.save)
    logger = utils.create_logger('global_logger', cfg.save + '/log.txt')
    PATH = '{}/{}'.format(cfg.save_path, args.model_name)
    print("Path test: {}".format(PATH))

    # Set cuda device & seed
    torch.cuda.set_device(cfg.gpu)
    np.random.seed(cfg.seed)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(cfg.seed)

    I = 10
    genotype_list, genotype_list_str = sampleGenotypeList()
    trainloader = cifar10(cfg.dataset_param.data_root, cfg.dataset_param.batch_size, cfg.dataset_param.num_workers)
    # genotype_list_str_in = []
    # genotype_list_str_in.append(genotype_list_str)
    net = models.model_search_entry(cfg)
    net = net.cuda()
    net_adv = AttackPGD(net, cfg.attack_param)
    # Load checkpoint.
    if not Debug:
        print('==> Resuming from checkpoint..')
        net_adv.load_state_dict(torch.load(cfg.resume_path))
    optimizer = torch.optim.SGD(net_adv.parameters(), lr=0.1, momentum=0.9)
    net_adv.train()
    
    # Using specific operation list
    # genotype_list = [3,3,3,3,3,1,3,1,2,3,1,1,1,3]
    # genotype_list = [1,1,2,3,0,1,2,0,0,3,3,0,3,1]
    cfg.netpara = sum(p.numel() for p in net_adv.parameters()) / 1e6
    logger.info("Genotypes: {}".format(genotype_list))
    logger.info('Config: {}'.format(pprint.pformat(cfg)))

    for epochs in range(I):
        logger.info("Epochs: {}".format(epochs))
        losses = utils.AverageMeter(0)
        top1 = utils.AverageMeter(0)
        top5 = utils.AverageMeter(0)
        for batch_idx, (inputs, targets) in enumerate(trainloader):


            inputs, targets = inputs.cuda(), targets.cuda()

            outputs, inputs_adv = net_adv(inputs, targets, genotype_list)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, 5))
            num = inputs.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            if batch_idx % cfg.report_freq == 0:
                logger.info(
                    'Train: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    .format(batch_idx, len(trainloader), loss=losses, top1=top1, top5=top5))

    final_loss = losses.avg
    final_top1 = top1.avg
    final_top5 = top5.avg

    logger.info(' * Info based on final epoch...')
    logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\t'.format(final_top1, final_top5, final_loss))

    try:
        torch.save(net_adv.state_dict(), PATH)
    except:
        PATH = '{}/{}'.format("./", args.model_name)
        torch.save(net_adv.state_dict(), PATH)

if __name__ == "__main__":
    finetune()
