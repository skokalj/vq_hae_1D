import torch
from torch.utils.data import DataLoader
import numpy as np
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
import lightning.pytorch as pl
        

def pca(X):
    _, stds, pcs = np.linalg.svd(X/np.sqrt(X.shape[0])) 

    return stds**2, pcs

if __name__ == '__main__':
    pl.seed_everything(1234567891)
    torch.set_default_dtype(torch.float32)

    #classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
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
    
    


    
    
   
    ds_test = ModulationsDataset(
        classes=classes,
        use_class_idx=True,
        level=0,
        num_iq_samples=1024,
        num_samples=int(num_classes*test_samples_per_class),
        include_snr=False
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



 
