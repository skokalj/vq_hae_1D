import torch
import os
from torch.utils.data import DataLoader
from utils import *
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from hae_lightning_1D import HQA
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='HQA Signal Processing Model')
    parser.add_argument('--EPOCHS', type=int, default=30, help='Number of epochs')
    parser.add_argument('--num_iq_samples', type=int, default=1024, help='Number of IQ samples')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--codebook_slots', type=int, default=256, help='Number of codebook slots')
    parser.add_argument('--codebook_dim', type=int, default=64, help='each codebook dimension')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--KL_coeff', type=float, default=0.1, help='KL coefficient')
    parser.add_argument('--CL_coeff', type=float, default=0.005, help='CL coefficient')
    parser.add_argument('--Cos_coeff', type=float, default=0.7, help='Cosine coefficient')
    parser.add_argument('--batch_norm', type=int, default=1, help='Use batch normalization')
    parser.add_argument('--codebook_init', type=str, default='normal', help='Codebook initialization method')
    parser.add_argument('--reset_choice', type=int, default=1, help='Reset choice')
    parser.add_argument('--cos_reset', type=int, default=1, help='Reset cos_coeff for further layers')
    parser.add_argument('--version', type=int, default=1, help='Which version of the checkpoint to run')
    parser.add_argument('--compress', type=int, default=2, help='2 or 4 times compression')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    num_classes = len(classes)
    training_samples_per_class = 4000
    valid_samples_per_class = 1000
    test_samples_per_class = 1000
    num_workers=32
    
    EPOCHS = args.EPOCHS
    num_iq_samples = args.num_iq_samples
    layers = args.layers
    codebook_slots = args.codebook_slots
    codebook_dim = args.codebook_dim
    num_res_blocks = args.num_res_blocks
    KL_coeff = args.KL_coeff
    CL_coeff = args.CL_coeff
    Cos_coeff = args.Cos_coeff
    batch_norm = args.batch_norm
    codebook_init = args.codebook_init
    reset_choice = args.reset_choice
    cos_reset = args.cos_reset
    version = args.version
    compress = args.compress
    
    
    codebook_visuals_dir = f'Codebook_visualizations/{compress}CNo_norm_Visuals_HAE_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer/version_{version}'


    batch_size = 64

    print(f'EPOCHS : {EPOCHS}')
    print(f'num_iq_samples : {num_iq_samples}')
    print(f'layers : {layers}')
    print(f'codebook_slots : {codebook_slots}')
    print(f'num_res_blocks : {num_res_blocks}')
    print(f'KL_coeff : {KL_coeff}')
    print(f'CL_coeff : {CL_coeff}')
    print(f'Cos_coeff : {Cos_coeff}')
    print(f'batch_norm : {batch_norm}')
    print(f'codebook_init : {codebook_init}')
    print(f'reset_choice : {reset_choice}')
    print(f'cos_reset : {cos_reset}')
    print(f'version : {version}')
    print(f'codebook_dim : {codebook_dim}')
    print(f'compression  : {compress}')
    
    data_transform = ST.Compose([
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
        shuffle=True,
        drop_last=True,
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    
    dl_test = DataLoader(
        dataset=ds_test,
        batch_size=16,
        shuffle=False,
        drop_last=True,
    )
    
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]

    model_save_path=os.path.join(f'Saved_models/HAE/', f"{compress}C_mod_No_Norm_HAE_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer_version_{version}.ckpt")
    

    for i in range(layers): 
        print(f'training Layer {i}')
        print('==============================================')
        if i == 0:
            hqa = HQA.init_bottom(
                input_feat_dim=2,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=0,
                KL_coeff = KL_coeff,
                CL_coeff = CL_coeff,
                Cos_coeff = Cos_coeff,
                batch_norm = batch_norm,
                codebook_init = codebook_init,
                reset_choice = reset_choice,
                output_dir = codebook_visuals_dir,
                codebook_slots = codebook_slots,
                codebook_dim = codebook_dim,
                layer = i,
                cos_reset = cos_reset,
                compress = compress
            )
            
        else:
            hqa = HQA.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=2,
                KL_coeff = KL_coeff,
                CL_coeff = CL_coeff,
                Cos_coeff = Cos_coeff,
                batch_norm = batch_norm,
                codebook_init = codebook_init,
                codebook_dim = codebook_dim,
                reset_choice = reset_choice,
                output_dir = codebook_visuals_dir,
                codebook_slots = codebook_slots,
                layer = i,
                cos_reset = cos_reset,
                compress = compress
            )
        print("loaded the encoder and decoder pretrained models")
        logger = TensorBoardLogger(f"tb_logs", name=f"{compress}C_mod_No_Norm_HAE_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer_version_{version}")
        
        trainer = pl.Trainer(max_epochs=EPOCHS, 
             logger=logger,  
             devices=1,
             accelerator = 'gpu',
             num_sanity_val_steps=0,
        )
        trainer.fit(hqa.float(), dl_train, dl_val)
        hqa_prev = hqa.eval()
        torch.save(hqa, model_save_path)  
        print(f'saved the model as {model_save_path}')
        print('==========================================')
    hqa_model = torch.load(model_save_path)
    #print(summary(hqa.to(device='cuda'),(2,1024),16),device='cuda')

