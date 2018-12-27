import cv2
from loss import dice_bce_loss
import numpy as np


def get_evaluation(mask, pred, list_metrics):
    output = {}
    if 'dice_coff':
        output['dice_coff'] = dice_bce_loss().soft_dice_coeff(mask, pred).item()
    if 'accuracy' in list_metrics:
        mask = mask.cpu().numpy()
        y_prob = pred.cpu().detach().numpy()
        y_prob[y_prob >= 0.5] = 1.
        y_prob[y_prob < 0.5] = 0.
        output['accuracy'] = 1 - np.sum(np.logical_xor(mask, y_prob).reshape(-1)) * 1.0 / mask.reshape(-1).shape[0]

    # if 'loss' in list_metrics:
    #        try:
    #            output['loss'] = metrics.log_loss(y_true, y_prob)
    #        except ValueError:
    #            output['loss'] = -1
    #    if 'confusion_matrix' in list_metrics:
    #        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output
