# # Example 06 - Modulation Classifier
# This notebook walks through a simple example of how to use the clean Sig53 dataset, load a pre-trained supported model, and evaluate the trained network's performance. Note that the experiment and the results herein are not to be interpreted with any significant value but rather serve simply as a practical example of how the `torchsig` dataset and tools can be used and integrated within a typical [PyTorch](https://pytorch.org/) and/or [PyTorch Lightning](https://www.pytorchlightning.ai/) workflow.

# ----
# ### Import Libraries
# First, import all the necessary public libraries as well as a few classes from the `torchsig` toolkit. An additional import from the `cm_plotter.py` helper script is also done here to retrieve a function to streamline plotting of confusion matrices.

from torchsig.transforms.target_transforms import DescToClassIndex
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from torchsig.transforms.transforms import (
    RandomPhaseShift,
    Normalize,
    ComplexTo2D,
    Compose,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
from torchsig.utils.cm_plotter import plot_confusion_matrix
from torchsig.datasets.modulations import ModulationsDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchsig.datasets import conf
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from utils import *
import pandas as pd
from load_datasets import load_sig 
from torchsig.utils.dataset import SignalFileDataset
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
from torchvision import transforms
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from scipy import interpolate
from scipy import signal as sp
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

args = parse_args()
EFF_EPOCHS = 15
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

print(f'EPOCHS : {EPOCHS}')
print(f'num_iq_samples : {num_iq_samples}')
print(f'layers : {layers}')
print(f'codebook_slots : {codebook_slots}')
print(f'codebook_dim : {codebook_dim}')
print(f'num_res_blocks : {num_res_blocks}')
print(f'KL_coeff : {KL_coeff}')
print(f'CL_coeff : {CL_coeff}')
print(f'Cos_coeff : {Cos_coeff}')  

print(f'batch_norm : {batch_norm}')
print(f'codebook_init : {codebook_init}')
print(f'reset_choice : {reset_choice}')
print(f'cos_reset : {cos_reset}')
print(f'version : {version}')
print(f'compression  : {compress}')



# ----
# ### Instantiate Modulation Dataset
classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
num_classes = len(classes)
training_samples_per_class = 4000
valid_samples_per_class = 1000
test_samples_per_class = 1000
num_workers=32
torch.set_default_dtype(torch.float32)
data_transform = ST.Compose([
    ST.Normalize(norm=np.inf),
    ST.ComplexTo2D(),
#    transforms.Lambda(lambda x: x.double()),
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

train_dataloader = DataLoader(
    dataset=ds_train,
    batch_size=16,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=ds_val,
    batch_size=16,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)
test_dataloader = DataLoader(
    dataset=ds_test,
    batch_size=64,
    num_workers=num_workers,
    shuffle=False,
    drop_last=True,
)
torch.set_default_dtype(torch.float32)
model_save_path=os.path.join("tb_logs", f"EfficientNet_Classes6_e{EPOCHS}.pt")
   

pretrained = False if not os.path.exists("tb_logs/efficientnet_b4.pt") else True

model = efficientnet_b4(
    pretrained=pretrained,
    path="tb_logs/efficientnet_b4.pt",
)



class ExampleNetwork(LightningModule):
    def __init__(self, model, data_loader, val_data_loader):
        super(ExampleNetwork, self).__init__()
        self.mdl: torch.nn.Module = model
        self.data_loader: DataLoader = data_loader
        self.val_data_loader: DataLoader = val_data_loader

        # Hyperparameters
        self.lr = 0.001
        self.batch_size = data_loader.batch_size

    def forward(self, x: torch.Tensor):
        return self.mdl(x.float())

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.forward(x.float())
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.data_loader

    def val_dataloader(self):
        return self.val_data_loader

    def training_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


example_model = ExampleNetwork(model, train_dataloader, val_dataloader)
example_model = example_model.float().to(device)


# Create and fit trainer for Efficient Net
trainer = Trainer(
    max_epochs=EFF_EPOCHS, 
    #callbacks=checkpoint_callback, 
    devices=1, accelerator="gpu"
)

PATH = f'6Class_efficientNet_epochs_noNorm{EFF_EPOCHS}.pt'

#torch.save(example_model.state_dict(),PATH)
print(f'model trained and saved as {PATH}')
example_model.load_state_dict(torch.load(PATH))
print('loaded efficient net ')

example_model = example_model.to(device=device).eval()

hae_save_path=os.path.join(f"Saved_models/HAE/", f"{compress}C_mod_No_Norm_HAE_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer_version_{version}.ckpt")


hqa_save_path=os.path.join(f"Saved_models/HQA/", f"{compress}C_mod_No_Norm_HQA_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer_version_{version}.ckpt")

hqa_model = torch.load(hae_save_path)

num_recons = 3

num_test_examples = len(ds_test)
for j in range(layers):
    for k in range(num_recons): 
        y_raw_preds = np.empty((num_test_examples, num_classes))
        y_preds = np.zeros((num_test_examples,))
        y_true = np.zeros((num_test_examples,))
        hqa=hqa_model[j]
        hqa = hqa.float().to(device)
        hqa.eval()
        
        for i in tqdm(range(0, num_test_examples)):
            # Retrieve data
            idx = i  # Use index if evaluating over full dataset
            
            data, label = ds_test[idx]
            #test_x = hqa.reconstruct(data)
            test_x = hqa.reconstruct(torch.from_numpy(np.expand_dims(data, 0)).float().to(device))
            # Infer
            #test_x = torch.from_numpy(np.expand_dims(test_x, 0)).float().to(device)
            pred_tmp = example_model.predict(test_x)
            pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
            # Argmax
            y_preds[i] = np.argmax(pred_tmp)
            # Store label
            y_true[i] = label
    
    
        acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
        plot_confusion_matrix(
            y_true,
            y_preds,
            classes=classes,
            normalize=True,
            title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
                acc * 100
            ),
            text=False,
            rotate_x_text=90,
            figsize=(16, 9),
        )
        
        confusionMatrix_save_path = f"confusionMatrices/HAE/{compress}C_mod_No_Norm_HAE_norm_06_efficientNet_classifier_with_{codebook_init}_BN{batch_norm}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{j+1}by{layers}layer_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_reconstructions_version_{version}/"


        path = Path(confusionMatrix_save_path)
        if os.path.exists(path):
            pass
        else:
            path.mkdir(parents=True)
            #plt.savefig(path)
        curr_path = os.getcwd()
        os.chdir(path)
        plt.savefig(f'Layer_{j+1}_reconstruction_{k+1}.png')
        os.chdir(curr_path)
        print(f"Layer {j+1}\nClassification Report: \nAccuracy {acc*100}")
        print(classification_report(y_true, y_preds))
        matplotlib.pyplot.close()



hqa_model = torch.load(hqa_save_path)

num_recons = 3

num_test_examples = len(ds_test)
for j in range(layers):
    for k in range(num_recons): 
        y_raw_preds = np.empty((num_test_examples, num_classes))
        y_preds = np.zeros((num_test_examples,))
        y_true = np.zeros((num_test_examples,))
        hqa=hqa_model[j]
        hqa = hqa.float().to(device)
        hqa.eval()
        
        for i in tqdm(range(0, num_test_examples)):
            # Retrieve data
            idx = i  # Use index if evaluating over full dataset
            
            data, label = ds_test[idx]
            #test_x = hqa.reconstruct(data)
            test_x = hqa.reconstruct(torch.from_numpy(np.expand_dims(data, 0)).float().to(device))
            # Infer
            #test_x = torch.from_numpy(np.expand_dims(test_x, 0)).float().to(device)
            pred_tmp = example_model.predict(test_x)
            pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
            # Argmax
            y_preds[i] = np.argmax(pred_tmp)
            # Store label
            y_true[i] = label
    
    
        acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
        plot_confusion_matrix(
            y_true,
            y_preds,
            classes=classes,
            normalize=True,
            title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
                acc * 100
            ),
            text=False,
            rotate_x_text=90,
            figsize=(16, 9),
        )
        
        confusionMatrix_save_path = f"confusionMatrices/HQA/{compress}C_mod_No_Norm_HQA_norm_06_efficientNet_classifier_with_{codebook_init}_BN{batch_norm}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{j+1}by{layers}layer_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_reconstructions_version_{version}/"


        path = Path(confusionMatrix_save_path)
        if os.path.exists(path):
            pass
        else:
            path.mkdir(parents=True)
            #plt.savefig(path)
        curr_path = os.getcwd()
        os.chdir(path)
        plt.savefig(f'Layer_{j+1}_reconstruction_{k+1}.png')
        os.chdir(curr_path)
        print(f"Layer {j+1}\nClassification Report: \nAccuracy {acc*100}")
        print(classification_report(y_true, y_preds))
        matplotlib.pyplot.close()




