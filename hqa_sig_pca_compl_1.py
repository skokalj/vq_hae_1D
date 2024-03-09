import torch
import os
from torch.utils.data import DataLoader
import numpy as np
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from hqa_lightning1D import HQA
from scipy import signal as sp
import matplotlib.pyplot as plt
import copy
        
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(X):
    _, stds, pcs = np.linalg.svd(X/np.sqrt(X.shape[0])) 

    return stds**2, pcs

if __name__ == '__main__':
    #torch.set_float32_matmul_precision('medium')
    #classes = ["bpsk","8pam","8psk","16qam","16pam","64qam","64psk","256qam","1024qam","16gmsk"]
    torch.set_default_dtype(torch.float32)
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    classes = ["32qam_cross"]
    num_classes = len(classes)
    training_samples_per_class = 4000
    valid_samples_per_class = 1000
    test_samples_per_class = 3000
    num_workers=32
    EPOCHS=100
    codebook_dim=64
    layers=5
    cos_coefficient=0.7
    
    

    data_transform = ST.Compose([
        #ST.RayleighFadingChannel((.01, .1), power_delay_profile=(1.0, .7, .1)),
        #ST.Normalize(norm=2, flatten=True),
        #ST.ComplexTo2D(),
    ])

    pl.seed_everything(1234567891)
    
   
    ds_test = ModulationsDataset(
        classes=classes,
        use_class_idx=True,
        level=0,
        num_iq_samples=1024,
        num_samples=int(num_classes*test_samples_per_class),
        include_snr=False,
        transform = data_transform
    )

  
    
    dl_test = DataLoader(
        dataset=ds_test,
        batch_size=3000,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    data, label = next(iter(dl_test))
    data = data.view(3000,-1)
    data=data.cpu().detach().numpy()

    # standardize the data
    #scaler = StandardScaler()
    #data_std = scaler.fit_transform(data)
    #cov_matrix = np.cov(data.T)

# calculate eigenvalues and eigenvectors
    stds, pcs = pca(data)
    #eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# sort the eigenvalues and eigenvectors
    #sorted_indices = np.argsort(eigenvalues)[::-1]
    #sorted_eigenvalues = eigenvalues[sorted_indices]
    explained_variance_ratio_= np.cumsum(stds)/np.sum(stds)
    #sorted_eigenvectors = eigenvectors[:, sorted_indices]    
# data_pca_99 contains the principal components that retain 99% of the variance
    #data_pca_99 = pca_mnist_99.fit_transform(data_std)
    print(explained_variance_ratio_)
    print(np.min(np.where(explained_variance_ratio_> 0.99)))



 
