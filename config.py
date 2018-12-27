import torch

class Config:
    CUDA = torch.cuda.is_available()
    SHAPE = 1500
    CROP_SHAPE = 1024
    ROOT = '/home/an/re/'
    NETWORK = 'unet'  # unet, res_unet, my_net

    TRAIN_INPUT = '/home/an/Downloads/mass_roads/train/sat'
    TRAIN_TARGET = '/home/an/Downloads/mass_roads/train/map'
    VALIDATION_INPUT = 0
    VALIDATION_TARGET = 0

    CHECKPOINT = 'checkpoints/checkpoint_{}.pt'.format(NETWORK)
    CHECKPOINT_NOOPTIM = 'checkpoints/checkpoint_nooptim_{}.pt'.format(NETWORK)
    LOG = 'logs/{}.log'.format(NETWORK)
    RESUME = True

    BATCH_SIZE = 2  # if CUDA: BATCH_SIZE_PER_CARD = BATCH_SIZE
    NEW_LR = 0.0002  # None
    ALPHA = 0.3

    STRATEGY = 2


class ConfigTest:
    CUDA = torch.cuda.is_available()
    TEST = '/home/n/projects/DeepGlobe-Road-Extraction-Challenge/Road_Extraction/'
    TARGET = '/home/n/projects/DeepGlobe-Road-Extraction-Challenge/submits/'

    CHECKPOINTS = 'checkpoints/checkpoint_my_net.pt'
    CROP_SHAPE = 1024
    THRESHHOLD = 0.3
    OVERLAP = 256
    NPY = ""

class ConfigForServer:

    CUDA = torch.cuda.is_available()
    SHAPE = 1500
    CROP_SHAPE = 1024
    ROOT = '/content/drive/My Drive/re/'
    NETWORK = 'my_net_3'  # unet, res_unet, my_net, my_net_3

    TRAIN_INPUT = '/content/drive/My Drive/mass_roads/train/sat'
    TRAIN_TARGET = '/content/drive/My Drive/mass_roads/train/map'
    VALIDATION_INPUT = 0
    VALIDATION_TARGET = 0

    CHECKPOINT = 'checkpoints/checkpoint_{}.pt'.format(NETWORK)
    CHECKPOINT_NOOPTIM = 'checkpoints/checkpoint_nooptim_{}.pt'.format(NETWORK)
    LOG = 'logs/{}.log'.format(NETWORK)
    RESUME = False
    NEW_LR = None # None
    ALPHA = 0.3

    STRATEGY = 1

    BATCH_SIZE = 4 # if CUDA: BATCH_SIZE_PER_CARD = BATCH_SIZE

class ConfigTestForServer:
    CUDA = torch.cuda.is_available()
    NETWORK = 'my_net_3'  # unet, res_unet, mynet, my_net_3
    TEST = '/content/drive/My Drive/mass_roads/test/sat/'
    TARGET = '/content/drive/My Drive/re/submits/{}/'.format(NETWORK)

    CHECKPOINT = '/content/drive/My Drive/re/checkpoints/checkpoint_{}.pt'.format(NETWORK)
    CROP_SHAPE = 1024
    THRESHHOLD = 0.5
    OVERLAP = 256
    NPY = ""