import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter

import cv2
import os
import os.path, time
import numpy
import tqdm
from datetime import datetime

from networks.my_net import MyNet
from networks.my_net_3 import MyNet3
from networks.unet_7layers import Unet
from networks.residual_unet import Residual_Unet
from loss import dice_bce_loss
from framework import FrameWork
from data import ImageFolder
if 'home' in os.getcwd():
    from config import Config
else:
    from config import ConfigForServer as Config

train_list = map(lambda x: x[:-5], os.listdir(Config.TRAIN_INPUT))
writer = SummaryWriter(Config.ROOT + "runs/{}".format(Config.NETWORK))

if Config.NETWORK == 'my_net':
    solver = FrameWork(MyNet, dice_bce_loss, 2e-4, alpha=Config.ALPHA)
elif Config.NETWORK == 'my_net_3':
    solver = FrameWork(MyNet3, dice_bce_loss, 2e-4, alpha=Config.ALPHA)
elif Config.NETWORK == 'unet':
    solver = FrameWork(Unet, dice_bce_loss, 2e-4, alpha=Config.ALPHA)
else:
    solver = FrameWork(Residual_Unet, dice_bce_loss, 2e-4, alpha=Config.ALPHA)
if Config.CUDA:
    batchsize = torch.cuda.device_count() * Config.BATCH_SIZE
else:
    batchsize = Config.BATCH_SIZE
if Config.RESUME:
    if time.ctime(os.path.getmtime(Config.ROOT + Config.CHECKPOINT_NOOPTIM)) > time.ctime(os.path.getmtime(Config.ROOT + Config.CHECKPOINT)):
        ck_path = Config.ROOT + Config.CHECKPOINT_NOOPTIM
    else:
        ck_path = Config.ROOT + Config.CHECKPOINT
    epoch_th, lr, train_epoch_best_loss, no_optim = solver.load(ck_path)
    epoch_th += 1
else:
    epoch_th, lr, train_epoch_best_loss, no_optim = [1, 2e-4, 100., 0]

if Config.NEW_LR:
    solver.update_lr(Config.NEW_LR, mylog=Config.ROOT + Config.LOG)
    epoch_th, lr, no_optim = [1, Config.NEW_LR, 0]



if epoch_th == 1:
    writer.add_graph(solver.net, torch.rand(1, 3, 1024, 1024))
writer.close()
dataset = ImageFolder(train_list, Config.TRAIN_INPUT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True)


tic = datetime.now()
total_epoch = 400

for epoch in range(epoch_th, total_epoch + 1):
    mylog = open(Config.ROOT + Config.LOG, 'a')
    writer = SummaryWriter(Config.ROOT + "runs/{}".format(Config.NETWORK))
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    train_epoch_accuracy = 0
    train_epoch_dicecoff = 0
    for img, mask in tqdm.tqdm(data_loader_iter):
        solver.set_input(img, mask)
        train_loss, train_metrics = solver.optimize()
        train_epoch_loss += train_loss
        train_epoch_accuracy += train_metrics["accuracy"]
        train_epoch_dicecoff += train_metrics["dice_coff"]
    train_epoch_loss /= len(data_loader_iter)
    train_epoch_accuracy /= len(data_loader_iter)
    train_epoch_dicecoff /= len(data_loader_iter)
    print >> mylog, '*'*80
    print >> mylog, datetime.now()
    print >> mylog, 'epoch:', epoch, '    time:', datetime.now() - tic
    print >> mylog, 'train_loss:', train_epoch_loss, '|best_loss', train_epoch_best_loss
    print >> mylog, 'accuracy:', train_epoch_accuracy
    print >> mylog, 'dice coff:', train_epoch_dicecoff
    print '*' * 80
    print datetime.now()
    print 'epoch:', epoch, '    time:', datetime.now() - tic
    print 'train_loss:', train_epoch_loss, '|best_loss', train_epoch_best_loss
    print 'accuracy:', train_epoch_accuracy
    print 'dice coff:', train_epoch_dicecoff

    writer.add_scalar('Train/Loss', train_epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', train_epoch_accuracy, epoch)
    writer.add_scalar('Train/DiceCoff', train_epoch_dicecoff, epoch)
    writer.close()
    mylog.close()

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
        solver.save(Config.ROOT + Config.CHECKPOINT_NOOPTIM, epoch, lr, train_epoch_best_loss, no_optim)
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save(Config.ROOT + Config.CHECKPOINT,  epoch, lr, train_epoch_best_loss, no_optim)
    if Config.STRATEGY == 1:
        if no_optim > 20:
            mylog = open(Config.ROOT + Config.LOG, 'a')
            print >> mylog, 'early stop at %d epoch' % epoch
            mylog.close()
            print 'early stop at %d epoch' % epoch
            break
        if no_optim > 10:
            if solver.old_lr < 5e-7:
                break
            solver.load(Config.ROOT +Config.CHECKPOINT)
            solver.update_lr(5.0, factor=True, mylog=Config.ROOT + Config.LOG)
    elif Config.STRATEGY == 2:
        if epoch % 100 == 0:
            solver.update_lr(5.0, factor=True, mylog=Config.ROOT + Config.LOG)
print 'Finish!'