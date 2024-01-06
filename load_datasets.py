# Author: Matt Williams
# Version: 12/26/2022

import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.dataset import SignalDataset
import torchsig.transforms as ST
from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer

from torch.utils.data import DataLoader, Subset
from utils import MNIST_TRANSFORM, EMNIST_TRANSFORM, MNIST_BATCH_SIZE, NUM_DATA_LOADER_WORKERS, \
RANDOM_SEED, FFT_MNIST_TRANSFORM
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split

# File paths for downloading mnist related datasets
MNIST_TRAIN_PATH = '/tmp/mnist'
MNIST_TEST_PATH = '/tmp/mnist_test_'
FASH_MNIST_TRAIN_PATH = '/tmp/fasion_mnist'
FASH_MNIST_TEST_PATH = '/tmp/fasion_mnist_test_'
EMNIST_TRAIN_PATH = '/tmp/emnist'
EMNIST_TEST_PATH = '/tmp/emnist_test_'
SIG_TRAIN_PATH = '/tmp/sig'
SIG_TEST_PATH = '/tmp/sig_test_'

def _make_train_valid_split(ds_train, len_ds_test):
    train_idxs, valid_idxs, _, _ = train_test_split(
            range(len(ds_train)),
            ds_train.targets,
            stratify=ds_train.targets,
            test_size= len_ds_test / len(ds_train), 
            random_state=RANDOM_SEED
        )
    ds_train = Subset(ds_train, train_idxs)
    ds_valid = Subset(ds_train, valid_idxs)
    
    return ds_train, ds_valid

def _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split, num_workers=NUM_DATA_LOADER_WORKERS, batch_size=MNIST_BATCH_SIZE):
    ds_valid = None
    if validate:
        ds_train, ds_valid = _make_train_valid_split(ds_train, len(ds_test))

    if return_tiled:
        ds_train = TiledDataset(ds_train, num_tiles, tile_split)
        ds_test = TiledDataset(ds_test, num_tiles, tile_split)
        if ds_valid:
            ds_valid = TiledDataset(ds_valid, num_tiles, tile_split)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_valid = None
    if ds_valid:
        dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    
    return dl_train, dl_valid, dl_test


def load_mnist(validate = False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, train=True, transform=MNIST_TRANSFORM)
    ds_test = MNIST(MNIST_TEST_PATH, download=True, train=False, transform=MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


def load_fft_mnist(validate = False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, train = True, transform=FFT_MNIST_TRANSFORM)
    ds_test = MNIST(MNIST_TEST_PATH, download=True, train=False, transform=FFT_MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


def load_fashion_mnist(validate=False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = FashionMNIST(FASH_MNIST_TRAIN_PATH, download=True, train = True, transform=MNIST_TRANSFORM)
    ds_test = FashionMNIST(FASH_MNIST_TEST_PATH, download=True, train = False, transform=MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


def load_emnist(split = "balanced", validate = False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = EMNIST(EMNIST_TRAIN_PATH, split = split, download = True, train = True, transform = EMNIST_TRANSFORM)
    ds_test = EMNIST(EMNIST_TEST_PATH, split = split, download = True, train = False, transform = EMNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)



def load_sig(classes = ["bpsk","8pam","8psk","16qam","16pam","64qam","64psk","256qam","1024qam","16gmsk"],
            level = 0, 
            include_snr = False, 
            dim_size = 32, 
            samples_per_class = 100, 
            validate = False, 
            return_tiled = False, 
            num_tiles = 2, 
            tile_split = "v",
            batch_size=MNIST_BATCH_SIZE):
    num_classes = len(classes)
    num_iq_samples = dim_size * dim_size

    data_transform = ST.Compose([
    ST.Normalize(norm=np.inf),
    ST.ComplexTo2D(),
    ])

# Seed the dataset instantiation for reproduceability
    pl.seed_everything(1234567891)

    ds_full = ModulationsDataset(
    classes=classes,
    use_class_idx=True, #False,
    level=level,
    num_iq_samples=1024,
    num_samples=int(num_classes*samples_per_class),
    include_snr=include_snr,
    transform = data_transform
    )
    ds_test_len = int(len(ds_full)/3)
    ds_test_indices = range(0, ds_test_len)
    ds_train_indices = range(ds_test_len, len(ds_full))
    ds_train, ds_test = Subset(ds_full, ds_train_indices), Subset(ds_full, ds_test_indices)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split, batch_size=batch_size, num_workers=0)

