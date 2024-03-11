import torch
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from pytorch_lightning import LightningModule
from torchsig.datasets.modulations import ModulationsDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
import lightning.pytorch as pl

from torch.utils.data import DataLoader
import numpy as np
from torchsig.datasets.modulations import ModulationsDataset
import torchsig.transforms as ST
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from tqdm import tqdm

def compression_ratio(layer_index):
    # CR_i = zeta * 2 ^ i
    return 1.37 * np.power(2, layer_index)

def plot_heirarchical_model(heirarchical_model, name, classifier, test_dataloader,ds_test):
    ds_test[0][0].size
    num_layers = len(heirarchical_model)
    compression_ratios = np.zeros(num_layers)
    accuracies = np.zeros(num_layers)
    compression_ratios = np.zeros(num_layers)
    accuracies = np.zeros(num_layers)
    for j in range(5):
        y_preds = np.zeros((len(ds_test),)) #0's of length dataset size
        y_true = np.zeros((len(ds_test),)) 
        hqa=heirarchical_model[j]
        hqa = hqa.float().to(device)
        hqa.eval()
        print(f"Layer {j}")
        for i in tqdm(range(0, len(ds_test))):
            idx = i  
            data, label = ds_test[idx]
            test_x = hqa.reconstruct(torch.from_numpy(np.expand_dims(data, 0)).float().to(device))
            pred_tmp = classifier.predict(test_x)
            pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
            y_preds[i] = np.argmax(pred_tmp)
            y_true[i] = label
    
    
        accuracy = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
        compression_ratios[j] = compression_ratio(j)
        accuracies[j] = accuracy
    return compression_ratios, accuracies
        

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    device = 'cuda'
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    num_classes = len(classes)
    training_samples_per_class = 4000
    valid_samples_per_class = 1000
    test_samples_per_class = 1000
    num_workers=32
    EPOCHS=100
    codebook_dim=64
    layers=5
    cos_coefficient=0.7
    test_datapoints = num_classes * test_samples_per_class
    
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
        shuffle=False,
        #drop_last=True,
    )


    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    MODEL_NAME = "[2024-02-12 16:31:32]HAE_64_5layers_0.70cos_lNonenorm.pt"   
    cb_init = 'normal'
    
    hqa_save_paths= ["checkpoints/[2024-03-01 16_33_53]HAE_64_5layers_0.70cos_lNonenorm.pt",
                    "checkpoints/[2024-03-03 00_00_12]HQA_64_5layers_0.70cos_lNonenorm.pt",
                    "checkpoints/DS_best_NO_KL_64_64.ckpt"]
    
    
    
    models = ["HAE",
              "HQARF",
              "HQARF_NO_KL"]
    '''
    models = ["HAE"]
    hqa_save_paths= ["[2024-02-12 16:31:32]HAE_64_5layers_0.70cos_lNonenorm.pt",
                    "Saved_models/HQA/best_NO_KL_64_64.ckpt",
                    "Saved_models/HQA/best_KL_64_64.ckpt"]
    '''
    model = efficientnet_b4(
        pretrained=False,
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
    
    PATH = f'6Class_efficientNet_epochs_noNorm15.pt'
    classifier = ExampleNetwork(model, dl_test, dl_test)
    classifier = classifier.float().to(device)
    classifier.load_state_dict(torch.load(PATH))
    
    plt.figure(figsize=(8, 4))
    for index, (name, checkpoint) in enumerate(zip(models, hqa_save_paths)):
        #plt.title("Accuracies")
        print(name)
        model = torch.load(checkpoint).to(device).float()
        compression_ratios, accuracies = plot_heirarchical_model(model, name, classifier.eval(), dl_test, ds_test)
        plt.plot(compression_ratios, accuracies, marker='o', label=name, linewidth=2)
        plt.xlabel("Compression Ratio", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xscale("log")
        plt.minorticks_off()
        compression_labels = [f'{compression_ratios[0]} (L0)', f'{compression_ratios[1]} (L1)', f'{compression_ratios[2]} (L2)', f'{compression_ratios[3]} (L3)', f'{compression_ratios[4]} (L4)']
        plt.xticks(compression_ratios, labels=compression_labels)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=10)
    
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize=12)
    plt.savefig(f"Accuracies/ds_All_accuracies.png", bbox_inches='tight')
