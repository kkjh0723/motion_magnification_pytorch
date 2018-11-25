# need to check

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from network import MagNet
from data_loader import ImageFromFolder
from utils import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch Deep Video Magnification')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=12, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--num_data', default=100000, type=int,
                    help='number of total data sample used for training (default: 100000)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt', default='ckpt', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')
parser.add_argument('--gpu',default=0, type=str, help='cuda_visible_devices')
parser.add_argument('--weight_reg1', default=1.0, type=float,
                    help='weight texture regularization loss  (default: 1.0)')
parser.add_argument('--weight_reg2', default=1.0, type=float,
                    help='weight shpae regularization loss  (default: 1.0)')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

losses_recon, losses_reg1, losses_reg2, losses_reg3 = [],[],[],[]

def main():
    global args
    global losses_recon, losses_reg1, losses_reg2, losses_reg3
    args = parser.parse_args()
    print(args)

    # create model
    model = MagNet()
    model  = torch.nn.DataParallel(model).cuda()
    print(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            losses_recon = checkpoint['losses_recon'] 
            losses_reg1 = checkpoint['losses_reg1']
            losses_reg2 = checkpoint['losses_reg2']
            losses_reg3 = checkpoint['losses_reg3']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # check saving directory
    ckpt_dir = args.ckpt
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir)

    # cudnn enable
    cudnn.benchmark = True

    # dataloader
    dataset_mag = ImageFromFolder('./../data/train', num_data=args.num_data, preprocessing=True) 
    data_loader = data.DataLoader(dataset_mag, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True)

    # loss criterion
    criterion = nn.L1Loss(size_average=True).cuda()

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                betas=(0.9,0.999),
                                weight_decay=args.weight_decay)


    # train model
    for epoch in range(args.start_epoch, args.epochs):
        loss_recon, loss_reg1, loss_reg2, loss_reg3 = train(data_loader, model, criterion, optimizer, epoch, args)
        
        # stack losses
        losses_recon.append(loss_recon)
        losses_reg1.append(loss_reg1)
        losses_reg2.append(loss_reg2)
        losses_reg3.append(loss_reg3)

        dict_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'losses_recon': losses_recon,
            'losses_reg1': losses_reg1,
            'losses_reg2': losses_reg2,
            'losses_reg3': losses_reg3,
        }
        
        # save checkpoints
        fpath = os.path.join(ckpt_dir, 'ckpt_e%02d.pth.tar'%(epoch))
        torch.save(dict_checkpoint, fpath)

def train(loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_recon = AverageMeter()
    losses_reg1 = AverageMeter() # texture loss
    losses_reg2 = AverageMeter() # shape loss
    losses_reg3 = AverageMeter()

    model.train()

    end = time.time()
    for i, (y, xa, xb, xc, amp_factor) in enumerate(loader):
        y = y.cuda(async=True)
        data_time.update(time.time() - end)

        # compute output
        amp_factor = amp_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        y_hat, rep_a, rep_b, rep_c = model(xa, xb, xc, amp_factor)
        #v_c, m_c = model.encoder(xc)
        v_a, m_a = rep_a # v: texture, m: shape
        v_b, m_b = rep_b
        v_c, m_c = rep_c

        # compute losses
        loss_recon = criterion(y_hat, y)
        loss_reg1 = args.weight_reg1 * L1_loss(v_c, v_a)
        loss_reg2 = args.weight_reg2 * L1_loss(m_c, m_b)
        loss_reg3 = 0.0
        loss = loss_recon + loss_reg1 + loss_reg2 + loss_reg3

        losses_recon.update(loss_recon.item()) 
        losses_reg1.update(loss_reg1.item()) 
        losses_reg2.update(loss_reg2.item()) 
        losses_reg3.update(loss_reg3)

        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LossR1 {loss_reg1.val:.4f} ({loss_reg1.avg:.4f})\t'
                  'LossR2 {loss_reg2.val:.4f} ({loss_reg2.avg:.4f})\t'
                  'LossR3 {loss_reg3.val:.4f} ({loss_reg3.avg:.4f})'.format(
                   epoch, i, len(loader), batch_time=batch_time, data_time=data_time, 
                   loss=losses_recon, loss_reg1=losses_reg1,
                   loss_reg2=losses_reg2, loss_reg3=losses_reg3))
    

    return losses_recon.avg, losses_reg1.avg, losses_reg2.avg, losses_reg3.avg    


def L1_loss(input, target):
    return torch.abs(input - target).mean()


 
if __name__ == '__main__':
    main()
