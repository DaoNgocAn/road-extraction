import torch
from scipy.stats.stats import mode
from torch.autograd import Variable as V

import cv2
import os
import os.path, time
import numpy as np
import tqdm
from datetime import datetime

from networks.my_net import MyNet
from networks.unet_7layers import Unet
from networks.residual_unet import Residual_Unet
from networks.my_net_4 import MyNet4

if 'home' in os.getcwd():
    from config import ConfigTest
else:
    from config import ConfigTestForServer as ConfigTest

BATCHSIZE_PER_CARD = 2


class TTAFrame():
    def __init__(self, net):
        if ConfigTest.CUDA:
            self.net = net().cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        else:
            self.net = net().cpu()
        self.net.eval()

    def test_one_img_from_path(self, path, evalmode=True):
        if ConfigTest.CUDA:
            batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
            if batchsize >= 8:
                return self.test_one_img_from_path_1(path)
            elif batchsize >= 4:
                return self.test_one_img_from_path_2(path)
            elif batchsize >= 2:
                return self.test_one_img_from_path_4(path)
        else:
            return self.test_one_img_from_path_8(path)

    def test_one_img_from_path_8(self, img):
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        def test_img(img):
            img1 = img[None]
            img2 = img1[:, ::-1]
            img3 = img1[:, :, ::-1]
            img4 = img2[:, :, ::-1]
            img1 = img1.transpose(0, 3, 1, 2)
            img2 = img2.transpose(0, 3, 1, 2)
            img3 = img3.transpose(0, 3, 1, 2)
            img4 = img4.transpose(0, 3, 1, 2)

            img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6))
            img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6))
            img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6))
            img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6))

            maska = self.net.forward(img1).squeeze().cpu().data.numpy()
            maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
            maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
            maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

            mask = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
            return mask

        mask = test_img(img)
        img90 = np.rot90(img)
        mask90 = test_img(img90)
        return mask + np.rot90(mask90)[::-1, ::-1]

    def test_cpu(self, img):
        img1 = img[None]
        img1 = img1.transpose(0, 3, 1, 2)
        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6))
        mask = self.net.forward(img1).squeeze().cpu().data.numpy()

        return mask

    def test_gpu(self, img):
        img1 = img[None]
        img1 = img1.transpose(0, 3, 1, 2)
        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        mask = self.net.forward(img1).squeeze().cpu().data.numpy()

        return mask

    def test_one_img_from_path_4(self, img):
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def load(self, path):
        checkpoint = torch.load(path)
        if ConfigTest.CUDA:
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
        self.net.eval()


def partition_image_1(img, crop_shape):
    h, w = img.shape[:2]
    times_h = h // crop_shape + 1
    times_w = w // crop_shape + 1
    if (crop_shape * times_h - h) % (times_h - 1) == 0:
        ConfigTest.overlap_h = (crop_shape * times_h - h) / (times_h - 1)
    else:
        '''
        viet sau
        '''
        ConfigTest.overlap_h = (crop_shape * times_h - h) / (times_h - 1)

    if (crop_shape * times_w - w) % (times_w - 1) == 0:
        ConfigTest.overlap_w = (crop_shape * times_w - w) / (times_w - 1)
    else:
        '''
        viet sau
        '''
        ConfigTest.overlap_w = (crop_shape * times_h - h) / (times_h - 1)

    imgs = []
    for i in range(times_h):
        row = []
        for j in range(times_w):
            try:
                row.append(
                    img[i * crop_shape - i * ConfigTest.overlap_h:(i + 1) * crop_shape - i * ConfigTest.overlap_h,
                    j * crop_shape - j * ConfigTest.overlap_w:(j + 1) * crop_shape - j * ConfigTest.overlap_w])
            except:
                continue
        imgs.append(row)
    return imgs


def partition_image_2(img, crop_shape, overlap):
    h, w = img.shape[:2]
    step = crop_shape - overlap
    imgs = []
    for i in range(0, h - crop_shape, step):
        row = []
        for j in range(0, w - crop_shape, step):
            row.append(img[i:i + crop_shape, j:j + crop_shape])
        imgs.append(row)
    return imgs


def partition_image(img, crop_shape, overlap, mode='full'):
    # mode = 'full' | 'partial'
    if mode == 'full':
        return partition_image_1(img, crop_shape)
    else:
        ConfigTest.overlap_h = ConfigTest.OVERLAP
        ConfigTest.overlap_w = ConfigTest.OVERLAP
        return partition_image_2(img, crop_shape, overlap)


def merge_with_overlap(list_array, axis, overlap):
    '''
    :param list_array:
    :param axis: = 1, hop tu` trai sang phai, doan overlap cong vao` chia 2, axis =0 tu` tren xuong duoi
    :param overlap: do dai overlap
    :return:
    '''
    img0 = list_array[0]
    if axis == 1:
        for i in range(1, len(list_array)):
            a = img0[:, -overlap:]
            b = list_array[i][:, :overlap]
            a[a < b] = b[b > a]
            img0[:, -overlap:] = a
            # img0[:, -overlap:] = (img0[:, -overlap:] + list_array[i][:, :overlap]) / 2
            img0 = np.concatenate([img0, list_array[i][:, overlap:]], axis=1)
    else:
        for i in range(1, len(list_array)):
            a = img0[-overlap:, :]
            b = list_array[i][:overlap, :]
            a[a < b] = b[b > a]
            img0[-overlap:, :] = a
            # img0[-overlap:, :] = (img0[-overlap:, :] + list_array[i][:overlap, :]) / 2
            img0 = np.concatenate([img0, list_array[i][overlap:, :]], axis=0)
    return img0


def predict_mask(path):
    img = cv2.imread(path)

    # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    def merge_mask(masks, id):
        rows = []
        for mask_row in masks:
            rows.append(merge_with_overlap(mask_row, axis=1, overlap=ConfigTest.overlap_h))

        mask = merge_with_overlap(rows, axis=0, overlap=ConfigTest.overlap_h)
        # np.save(npy + name[:-4] + '_{}.npy'.format(id), mask)
        return mask

    masks = [[] for _ in range(8)]
    for row in partition_image(img, ConfigTest.CROP_SHAPE, ConfigTest.OVERLAP, mode='full'):
        mask_row = [[] for _ in range(8)]
        for _img in row:
            # _img = _img.transpose(2,0,1)
            mask_row[0].append(solver.test_gpu(_img))
            mask_row[1].append(solver.test_gpu(_img[::-1])[::-1])
            mask_row[2].append(solver.test_gpu(_img[:, ::-1])[:, ::-1])
            mask_row[3].append(solver.test_gpu(_img[::-1, ::-1])[::-1, ::-1])
            _img = np.rot90(_img)
            mask_row[4].append(solver.test_gpu(_img))
            mask_row[5].append(solver.test_gpu(_img[::-1])[::-1])
            mask_row[6].append(solver.test_gpu(_img[:, ::-1])[:, ::-1])
            mask_row[7].append(solver.test_gpu(_img[::-1, ::-1])[::-1, ::-1])
            for i in range(4, 8):
                mask_row[i][-1] = np.rot90(mask_row[i][-1])[::-1, ::-1]
        for i in range(8):
            masks[i].append(mask_row[i])
    mm = []
    for i in range(8):
        mm.append(merge_mask(masks[i], i))
    m = np.zeros_like(mm[1])
    c = mm[0]
    for i in range(1, 8):
        c = c + mm[i]

    m[c > ConfigTest.THRESHHOLD * 8] = 255
    m[c <= ConfigTest.THRESHHOLD * 8] = 0
    cv2.imwrite(target + name[:-5] + '_mask.png', m.astype(np.uint8))


# source = 'dataset/test/'
source = ConfigTest.TEST
val = os.listdir(source)
if ConfigTest.NETWORK == 'my_net':
    solver = TTAFrame(MyNet)
elif ConfigTest.NETWORK == 'unet':
    solver = TTAFrame(Unet)
elif ConfigTest.NETWORK == 'my_net_4':
    solver = TTAFrame(MyNet4)
else:
    solver = TTAFrame(Residual_Unet)

print "loading" + ConfigTest.NETWORK
solver.load(ConfigTest.CHECKPOINT)
tic = datetime.now()
target = ConfigTest.TARGET
if not os.path.isdir(target):
    os.makedirs(target)
npy = ConfigTest.NPY

# os.mkdir(target)
for i, name in enumerate(val):
    print name, '    ', (datetime.now() - tic)
    # mask = solver.test_one_img_from_path(source + name)
    # mask[mask > 4.0] = 255
    # mask[mask <= 4.0] = 0
    # mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
    # cv2.imwrite(target + name[:-7] + 'mask.png', mask.astype(np.uint8))
    predict_mask(source + name)
    print "Done"