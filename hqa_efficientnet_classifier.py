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


EPOCHS = 4

# ----
# ### Instantiate Modulation Dataset
classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
num_classes = len(classes)
training_samples_per_class = 4000
valid_samples_per_class = 1000
test_samples_per_class = 1000
num_workers=32

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
    num_iq_samples=4096,
    num_samples=int(num_classes*training_samples_per_class),
    include_snr=False,
    transform = data_transform
)
ds_val = ModulationsDataset(
    classes=classes,
    use_class_idx=True,
    level=0,
    num_iq_samples=4096,
    num_samples=int(num_classes*valid_samples_per_class),
    include_snr=False,
    transform = data_transform
)

ds_test = ModulationsDataset(
    classes=classes,
    use_class_idx=True,
    level=0,
    num_iq_samples=4096,
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
   


# ----
# ### Format Dataset for Training
# Next, the datasets are then wrapped as `DataLoaders` to prepare for training.

# ### Instantiate Supported TorchSig Model
# Below, we load a pretrained EfficientNet-B4 model, and then conform it to a PyTorch LightningModule for training.
pretrained = False if not os.path.exists("tb_logs/efficientnet_b4.pt") else True

model = efficientnet_b4(
    pretrained=pretrained,
    path="tb_logs/efficientnet_b4.pt",
)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)


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


# ----
# ### Train the Model
# To train the model, we first create a `ModelCheckpoint` to monitor the validation loss over time and save the best model as we go. The network is then instantiated and passed into a `Trainer` to kick off training.

# Setup checkpoint callbacks
checkpoint_filename = "{}/tb_logs/checkpoints/checkpoint".format(os.getcwd())
checkpoint_callback = ModelCheckpoint(
    filename=checkpoint_filename,
    save_top_k=True,
    monitor="val_loss",
    mode="min",
)
logger = TensorBoardLogger('tb_logs/efficientNet',name=f'efficientNet_1D_{EPOCHS}')

# Create and fit trainer
trainer = Trainer(
    max_epochs=EPOCHS, callbacks=checkpoint_callback, devices=1, accelerator="gpu",logger=None
)
trainer.fit(example_model, train_dataloader, val_dataloader) 


# ----
# ### Evaluate the Trained Model
# After the model is trained, the checkpoint's weights are loaded into the model and the model is put into evaluation mode. The validation set is looped through, inferring results for each example and saving the predictions and the labels. Finally, the labels and predictions are passed into our confusion matrix plotting function to view the results and also passed into the `sklearn.metrics.classification_report` method to print metrics of interest.
#checkpoint_filename = "{}/tb_logs/checkpoints/checkpoint".format(os.getcwd())
# Load best checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(
    checkpoint_filename + ".ckpt", map_location=lambda storage, loc: storage
)
example_model.load_state_dict(checkpoint["state_dict"])
example_model = example_model.to(device=device).eval()

# Infer results over test set
num_test_examples = len(ds_test)
y_raw_preds = np.empty((num_test_examples, num_classes))
y_preds = np.zeros((num_test_examples,))
y_true = np.zeros((num_test_examples,))

for i in tqdm(range(0, num_test_examples)):
    # Retrieve data
    idx = i  # Use index if evaluating over full dataset
    data, label = ds_test[idx]
    # Infer
    data = torch.from_numpy(np.expand_dims(data, 0)).float().to(device)
    pred_tmp = example_model.predict(data)
    pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
    # Argmax
    y_preds[i] = np.argmax(pred_tmp)
    # Store label
    y_true[i] = label


acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
plot_confusion_matrix(
    y_true,
    y_preds,
    classes=classes, #class_list,
    normalize=True,
    title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
        acc * 100
    ),
    text=False,
    rotate_x_text=90,
    figsize=(16, 9),
)
plt.savefig("06_mod_classifier.png")

print("Classification Report:")
print(classification_report(y_true, y_preds))
hqa_save_path=os.path.join("tb_logs", 'HQA3_Sig_1D_4096_2Res_1R_KL_C_Classes6_e4.pt')#HQA_Sig_1D_4096_2Res_1R_KL_C_Classes6_e{EPOCHS}.pt")

hqa_model = torch.load(hqa_save_path)

num_test_examples = len(ds_test)
#for j in range(2): #changed from five for  faster evaluation
for j in range(1): #changed from five for  faster evaluation
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
    plt.savefig(f"06_mod_classifier{j}.png")

    print("Classification Report:")
    print(classification_report(y_true, y_preds))


