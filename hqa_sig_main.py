import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from torchvision.utils import make_grid
import numpy as np
#from hqa_sig import *
import pandas as pd
from load_datasets import load_sig 
from torchsig.utils.dataset import SignalFileDataset
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from hqa_lightning_1D import HQA

from scipy import interpolate
from scipy import signal as sp


#from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer

if __name__ == '__main__':
    #torch.set_float32_matmul_precision('medium')
    torch.set_default_dtype(torch.float32)
    #classes = ["bpsk","8pam","8psk","16qam","16pam","64qam","64psk","256qam","1024qam","16gmsk"]
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    num_classes = len(classes)
    training_samples_per_class = 4000
    valid_samples_per_class = 1000
    test_samples_per_class = 1000
    num_workers=15
    EPOCHS=10
    num_iq_samples = 4096
    layers = 5
    num_res_blocks = 2
    KL_coeff =  0.2
    CL_coeff = 0.001
    Cos_coeff = 0.7
    torch.set_default_dtype(torch.float64)
    batch_size = 32


    data_transform = ST.Compose([
        ST.Normalize(norm=np.inf),
        ST.ComplexTo2D(),
    ])

    pl.seed_everything(1234567891)
    
    ds_train = ModulationsDataset(
        classes=classes,
        use_class_idx=True,
        level=0,
        num_iq_samples=num_iq_samples,
        num_samples=int(num_classes*training_samples_per_class),
        include_snr=False,
        transform = data_transform
    )

    ds_val = ModulationsDataset(
        classes=classes,
        use_class_idx=True,
        level=0,
        num_iq_samples=num_iq_samples,
        num_samples=int(num_classes*valid_samples_per_class),
        include_snr=False,
        transform = data_transform
    )    

    ds_test = ModulationsDataset(
        classes=classes,
        use_class_idx=True,
        level=0,
        num_iq_samples=num_iq_samples,
        num_samples=int(num_classes*test_samples_per_class),
        include_snr=False,
        transform = data_transform
    )

    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        #num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        #num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )
    
    dl_test = DataLoader(
        dataset=ds_test,
        batch_size=16,
        #num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )
    
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    model_save_path=os.path.join('Saved_models', f"HQA_Sig_1D_iq{num_iq_samples}_{layers}layer_res{num_res_blocks}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}.ckpt")
    
    for i in range(layers): #changed from five for  faster evaluation
        print(f'training Layer {i}')
        print('==============================================')
        if i == 0:
            hqa = HQA.init_bottom(
                input_feat_dim=2,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks,
                KL_coeff = KL_coeff,
                CL_coeff = CL_coeff,
                Cos_coeff = Cos_coeff,

            )
        else:
            hqa = HQA.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks,
                KL_coeff = KL_coeff,
                CL_coeff = CL_coeff,
                Cos_coeff = Cos_coeff,
            )
        logger = TensorBoardLogger("tb_logs/HQA_1D", name=f"HQA_Sig_1D_iq{num_iq_samples}_{i+1}th_layer_res{num_res_blocks}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}")

        trainer = pl.Trainer(max_epochs=EPOCHS, 
             logger=logger,  
             devices=[0],
             accelerator = 'gpu',
             num_sanity_val_steps=0
        )

        #trainer.fit(model=hqa, train_dataloaders=dl_train, val_loaders=dl_val)
        trainer.fit(hqa.double(), dl_train, dl_val)
        hqa_prev = hqa
        torch.save(hqa, model_save_path)  
        print(f'saved the model as {model_save_path}')
        print('==========================================')
    hqa_model = torch.load(model_save_path)

    for i in range(layers): #changed from five for  faster evaluation
        hqa=hqa_model[i]
        test_x, lab = next(iter(dl_test))
        hqa.eval()
        test_y = hqa.reconstruct(test_x)
        test_y = test_y.detach().cpu().numpy()
        batch_size= test_y.shape[0]
        figure1 = plt.figure()
        for k in range(batch_size):
            test_xiq = test_x.detach().cpu().numpy()[k,:,:]
            x=test_xiq[0,:]+ 1j*test_xiq[1,:]
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                k+1,
            )
            _, _, spectrogram = sp.spectrogram(
                x=x,
                fs=1.0,
                window=sp.windows.tukey(1024, 0.25),
                nperseg=1024,
                return_onesided=False,
                nfft=4096
            )
            spectrogram = 20 * np.log10(np.fft.fftshift(np.abs(spectrogram) + np.finfo(float).eps, axes=0))
            plt.imshow(
                spectrogram,
                vmin=np.min(spectrogram[spectrogram != -np.inf]),
                vmax=np.max(spectrogram[spectrogram != np.inf]),
                aspect="auto",
                cmap="jet",
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(str(lab[k]))
        figure1.savefig(f'Visuals/spectr2oKL{i}{k}_e{EPOCHS}.png')            
        figure2 = plt.figure(2)    
        for k in range(batch_size):
            test_yiq=test_y[k,:,:]
            x_hat=test_yiq[0,:]+ 1j*test_yiq[1,:]
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                k+1,
            )
            _, _, spectrogram = sp.spectrogram(
                x=x_hat,
                fs=1.0,
                window=sp.windows.tukey(1024, 0.25),
                nperseg=1024,
                return_onesided=False,
                nfft=4096
            )
            spectrogram = 20 * np.log10(np.fft.fftshift(np.abs(spectrogram) + np.finfo(float).eps, axes=0))
            plt.imshow(
                spectrogram,
                vmin=np.min(spectrogram[spectrogram != -np.inf]),
                vmax=np.max(spectrogram[spectrogram != np.inf]),
                aspect="auto",
                cmap="jet",
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(str(lab[k]))
        figure2.savefig(f'Visuals/spectr2oKL_hat{i}{k}_e{EPOCHS}.png')
        figure3 = plt.figure(3) 
        for k in range(batch_size):
            test_xiq = test_x.detach().cpu().numpy()[k,:,:]
            x=test_xiq[0,:]+ 1j*test_xiq[1,:]
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                k+1,
            )
            plt.plot(np.real(x))
            plt.plot(np.imag(x))
            plt.xticks([])
            plt.yticks([])
            plt.title(str(lab[k]))
        figure3.savefig(f'Visuals/iq2oKL{i}{k}_e{EPOCHS}.png')
        figure4 = plt.figure(4) 
        for k in range(batch_size):
            test_yiq=test_y[k,:,:]
            x_hat=test_yiq[0,:]+ 1j*test_yiq[1,:]
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                k+1,
            )
            plt.plot(np.real(x_hat))
            plt.plot(np.imag(x_hat))
            plt.xticks([])
            plt.yticks([])
            plt.title(str(lab[k]))
        figure4.savefig(f'Visuals/iq2oKL_hat{i}{k}_e{EPOCHS}.png')
    plt.close('all')
