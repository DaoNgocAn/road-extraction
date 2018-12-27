import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
import os

from utils import get_evaluation

if 'home' in os.getcwd():
    from config import Config
else:
    from config import ConfigForServer as Config


class FrameWork():
    def __init__(self, net, loss, lr=2e-4, evalmode=False, **kwargs):
        if Config.CUDA:
            self.net = net(**kwargs).cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        else:
            self.net = net(**kwargs)

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss(kwargs['alpha'])
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        if Config.CUDA:
            img = V(torch.Tensor(img).cuda())
        else:
            img = V(torch.Tensor(img).cpu())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def forward(self, volatile=False):
        if Config.CUDA:
            self.img = V(self.img.cuda(), volatile=volatile)
            if self.mask is not None:
                self.mask = V(self.mask.cuda(), volatile=volatile)
        else:
            self.img = V(self.img.cpu(), volatile=volatile)
            if self.mask is not None:
                self.mask = V(self.mask.cpu(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        training_metrics = get_evaluation(self.mask, pred,
                                          list_metrics=["accuracy", "dice_coff"])
        return loss.item(), training_metrics

    def save(self, path, epoch=1, lr=2e-4, train_epoch_best_loss=100., no_optim=0):
        torch.save({'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'learning_rate': lr,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': train_epoch_best_loss,
                    'no_optim': no_optim}, path)

    def load(self, path, evalMode=False):
        checkpoint = torch.load(path)
        if Config.CUDA:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            state_dict = torch.load(path, map_location='cpu')
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            self.net.load_state_dict(new_state_dict)
        if evalMode:
            self.net.eval()
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.net.train()

        return checkpoint['epoch'], checkpoint['learning_rate'], checkpoint['best_loss'], checkpoint['no_optim']

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        with open(mylog, 'a') as f:
            print >> f, 'update learning rate: %f -> %f' % (self.old_lr, new_lr)
        print 'update learning rate: %f -> %f' % (self.old_lr, new_lr)
        self.old_lr = new_lr