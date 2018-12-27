import os
from config import ConfigTest
os.chdir("/content/drive/My Drive/re/")
NETWORK = ConfigTest.NETWORK  # unet, res_unet, mynet
TEST = ConfigTest.TEST
TARGET = ConfigTest.TARGET

import cv2
import os.path
import numpy as np
import tqdm

def iou(mask, prediction):
    intersection = np.logical_and(mask, prediction)
    union = np.logical_or(mask, prediction)
    iou_score = np.sum(intersection) * 1.0 / np.sum(union)
    return iou_score

total = 0
val = os.listdir(TARGET)
for name in tqdm.tqdm(val):
    mask = cv2.imread(os.path.join(TEST, '{}.tif').format(name[:-9]), cv2.IMREAD_GRAYSCALE) / 255
    mask = np.array(mask, np.int8)
    prediction = cv2.imread(os.path.join(TARGET, name), cv2.IMREAD_GRAYSCALE) / 255
    prediction = np.array(prediction, np.int8)

    total += iou(mask, prediction)

print ""
print NETWORK, " : ", total/len(val)