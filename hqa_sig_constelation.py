import torch
import os
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from hqa_lightning_1D import HQA
from scipy import signal as sp
import matplotlib.pyplot as plt
import copy

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    num_classes = len(classes)
    training_samples_per_class = 4000
    valid_samples_per_class = 1000
    test_samples_per_class = 20
    num_workers=32
    EPOCHS=100
    codebook_dim=64
    layers=5
    cos_coefficient=0.7
    
    data_transform = ST.Compose([
        ST.ComplexTo2D(),
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
        batch_size=120,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )


    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    MODEL_NAME = "[2024-02-12 16:31:32]HAE_64_5layers_0.70cos_lNonenorm.pt"     
    #hqa_save_paths= ["Saved_models/HQA/best_NO_KL_64_64.ckpt"]
    #models = ["best_NO_KL_64_64"]
    
    hqa_save_paths= ["[2024-02-12 16:31:32]HAE_64_5layers_0.70cos_lNonenorm.pt",
                    "Saved_models/HQA/best_NO_KL_64_64.ckpt",
                    "Saved_models/HQA/best_KL_64_64.ckpt"]
    cb_init = 'normal'
    
    models = ["HAE",
              "HQA_NO_KL",
              "HQARF"]
    
    for model,hqa_save_path in zip(models,hqa_save_paths):
        hqa_model = torch.load(hqa_save_path).float()

        for target in range(6):
            indices = [i for i in range(len(ds_test)) if ds_test[i][1] == target]
            name = [name for name, num in ds_test.class_dict.items() if num == target][0]
            my_subset = Subset(ds_test, indices)
            loader = DataLoader(my_subset, batch_size=len(indices))
            test_x, _ = next(iter(loader))
            
            for ii in [0,1,5]:
                plt.figure(figsize=(5, 5))
                if ii == 0 :
                    x_i = []
                    x_q = []
                    for k in range(len(indices)):
                        test_xiq = test_x.detach().cpu().numpy()[k,:,:]
                        x=test_xiq[0,:]+ 1j*test_xiq[1,:]
                        x_i.append(np.real(x))
                        x_q.append(np.imag(x))
                    
                    plt.plot(x_i,x_q,'r',linestyle="",marker="o")
                    plt.axis('off')  # Turn off axis
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(-1, 1)
                    plt.ylim(-1, 1)
                    x_i = []
                    x_q = []
                else:
                    hqa = hqa_model[ii-1]
                    hqa.eval()
                    if model == 'HAE':
                        test_y = hqa.reconstruct(test_x)
                    else:
                        z_q, cc = hqa.codebook.quantize(hqa.encode(test_x))
                        test_y = hqa.reconstruct_from_codes(cc)
                    test_y = test_y.detach().cpu().numpy()
                    
                    x_i = []
                    x_q = []
                    for k in range(len(indices)):
                        test_xiq = test_y[k,:,:]
                        x = test_xiq[0,:] + 1j * test_xiq[1,:]
                        x_i.append(np.real(x))
                        x_q.append(np.imag(x))
                    plt.plot(x_i, x_q, 'r', linestyle="", marker="o")
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(-1, 1)
                    plt.ylim(-1, 1)
                    plt.axis('off')  # Turn off axis
                
                plt.tight_layout()
                plt.savefig(f'Constellations/{name}_layer{ii}_{model}.jpg')
                plt.close()
