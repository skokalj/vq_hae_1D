import torch
import random
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from PIL import Image
from pathlib import Path
from torchvision.utils import make_grid
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import torchvision.transforms.functional as TF
import scipy.fftpack as fp

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
DISTRIBUTED = False
if DISTRIBUTED:
    device = 0
LAYER_NAMES = ["Layer 0", "Layer 1", "Layer 2", "Layer 3", "Layer 4 Final"]
RECON_ROOT_NAMES = ["data_original", "data_jpg", "data_recon_0", "data_recon_1",
                    "data_recon_2", "data_recon_3", "data_recon_4"]

HQA_MNIST_MODEL_NAME = "hqa_mnist_model"
HQA_FASH_MNIST_MODEL_NAME = "hqa_fash_mnist_model"
HQA_EMNIST_MODEL_NAME = "hqa_emnist_model"
HQA_TILED_MNIST_MODEL_NAME = "hqa_tiled_mnist_model"
HQA_TILED_FASH_MNIST_MODEL_NAME = "hqa_tiled_fash_mnist_model"
HQA_TILED_EMNIST_MODEL_NAME = "hqa_tiled_emnist_model"

HQA_NUM_RES_BLOCKS=2
HQA_SIG_MODEL_NAME=f"hqa_sig1D_4K_resnet{HQA_NUM_RES_BLOCKS}_model4096"
HQA_SIG_SAVE_PATH=f"models/hqa_sig1D_4K_resnet{HQA_NUM_RES_BLOCKS}_model4096.pt"
IMG_SIG_DIR_PATH=f"hqa_sig1D_4K_resnet{HQA_NUM_RES_BLOCKS}_image4096"

WORLD_SIZE=2

CWD = os.path.abspath(os.getcwd())

#Saved data file paths (original data and constructions)
IMG_DIR_PATH = os.path.join(CWD, "data")
IMG_MNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "MNIST")
IMG_FASH_MNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "Fashion_MNIST")
IMG_EMNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "EMNIST")
IMG_TILED_MNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "Tiled_MNIST")
IMG_TILED_FASH_MNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "Tiled_Fashion_MNIST")
IMG_TILED_EMNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "Tiled_EMNIST")

#Model relatd file paths
MODELS_DIR = os.path.join(CWD, "models")
LOG_DIR = os.path.join(MODELS_DIR, "log")
HQA_MNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_mnist_model.pt")
HQA_FASH_MNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_fash_mnist_model.pt")
HQA_EMNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_emnist_model.pt")
HQA_TILED_MNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_tiled_mnist_model.pt")
HQA_TILED_FASH_MNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_tiled_fash_mnist_model.pt")
HQA_TILED_EMNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_tiled_emnist_model.pt")

LENET_MNIST_PATH = os.path.join(MODELS_DIR, "lenet_mnist.pt")
LENET_FASH_MNIST_PATH = os.path.join(MODELS_DIR, "lenet_fash_mnist.pt")
LENET_EMNIST_PATH = os.path.join(MODELS_DIR, "lenet_emnist.pt")

#GELU Experiment
HQA_MNIST_GELU_MODEL_NAME = "hqa_mnist_model_GELU"
HQA_MNIST_GELU_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_mnist_model_GELU.pt")
IMG_MNIST_GELU_DIR_PATH = os.path.join(IMG_DIR_PATH, "MNIST_GELU")

#FFT Experiment
HQA_MNIST_FFT_MODEL_NAME = "hqa_mnist_model_FFT"
HQA_MNIST_FFT_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_mnist_model_FFT.pt")
IMG_MNIST_FFT_DIR_PATH = os.path.join(IMG_DIR_PATH, "MNIST_FFT")


ACCURACY_OUTPUT_FILE = os.path.join(CWD, "classification_accuracies.csv")
ACCURACY_FILE_COLS = ["Model", "Dataset", "Reconstruction", "Attack", "Average Accuracy"] 
VISUAL_DIR = os.path.join(CWD, "Visuals")
MNIST_BATCH_SIZE = 512
NUM_DATA_LOADER_WORKERS = 4
RANDOM_SEED = 42

VECT_PERS_OUTPUT_FILE = os.path.join(CWD, "vectorized_persistences.csv")
VECT_PERS_OUTPUT_COLS = ["Model", "Dataset", "Label", "Reconstruction", "Attack", "Vectorized Persistence"]

MNIST_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

def fft_image(img):
    data = np.array(img)
    fftdata = fp.rfft(fp.rfft(data, axis = 0), axis = 1)
    remmax = lambda x : x / x.max()
    remmin = lambda x : x - np.amin(x, axis = (0,1), keepdims=True)
    toint8 = lambda x: (remmax(remmin(x)) * (256 - 1e-4)).astype(int)
    img = Image.new('L', toint8(fftdata).shape[1::-1])
    return img

FFT_MNIST_TRANSFORM = transforms.Compose([
    transforms.Lambda(fft_image),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])
    

def hflip_image(img): 
    return TF.hflip(img)


def rotate_image(img):
    return TF.rotate(img, -90)


EMNIST_TRANSFORM = transforms.Compose([
    transforms.Lambda(rotate_image),
    transforms.Lambda(hflip_image),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])


IMG_FOLDER_TRANFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def add_accuracy_results(model_name, dataset_name, reconstruction, attack_name, avg_accuracy):

    results_df = pd.read_csv(ACCURACY_OUTPUT_FILE, index_col = False) \
                if os.path.exists(ACCURACY_OUTPUT_FILE) \
                else pd.DataFrame(columns=ACCURACY_FILE_COLS)

    row_dict = {"Model" : [model_name], 
                "Dataset" : [dataset_name], 
                "Reconstruction" : [reconstruction],
                "Attack" : [attack_name],
                "Average Accuracy" : [avg_accuracy]}
    row_df = pd.DataFrame(row_dict, columns=ACCURACY_FILE_COLS)
    results_df = pd.concat([results_df, row_df], ignore_index=True)

    results_df.to_csv(ACCURACY_OUTPUT_FILE, index = False)


def add_vectorized_persistence(model_name, ds_name, label, reconstruction, attack_name, vector_persistence):


    vec_pers_df = pd.read_csv(VECT_PERS_OUTPUT_FILE, index_col=False) \
                if os.path.exists(VECT_PERS_OUTPUT_FILE) \
                else pd.DataFrame(columns = VECT_PERS_OUTPUT_COLS)

    row_dict = {"Model" : [model_name],
                "Dataset" : [ds_name],
                "Label" : [label],
                "Reconstruction" : [reconstruction],
                "Attack": [attack_name],
                "Vectorized Persistence" : [vector_persistence]}
    
    row_df = pd.DataFrame(row_dict, columns=VECT_PERS_OUTPUT_COLS)
    vec_pers_df = pd.concat([vec_pers_df, row_df], ignore_index=True)

    vec_pers_df.to_csv(VECT_PERS_OUTPUT_FILE, index = False)


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index] * 255, mode = "F")
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)


def set_seeds(seed=42, fully_deterministic=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def show_image(im_data, scale=1):
    dpi = matplotlib.rcParams['figure.dpi']
    height, width = im_data.shape
    figsize = scale * width / float(dpi), scale * height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    ax.imshow(im_data, vmin=0, vmax=1, cmap='gray')
    plt.show()
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)


# TRAINING
def show_recon(img, *models):
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(10 * len(models), 5))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, model in enumerate(models):
        model.eval()
        img_ = img.unsqueeze(0).unsqueeze(0)
        recon = model.reconstruct(img_).squeeze()
        output = np.hstack([img.cpu(), np.ones([img.shape[0], 1]), recon.cpu(), np.ones([img.shape[0], 1]), np.abs((img-recon).cpu())])
        axes[i].imshow(output, vmin=0, vmax=1, cmap='gray')
        model.train()


#TRAINING
def get_bit_usage(indices):
    """ Calculate bits used by latent space vs max possible """
    num_latents = indices.shape[0] * indices.shape[1] * indices.shape[2]
    avg_probs = F.one_hot(indices).float().mean(dim=(0, 1, 2))
    highest_prob = torch.max(avg_probs)
    bits = (-(avg_probs * torch.log2(avg_probs + 1e-10)).sum()) * num_latents
    max_bits = math.log2(256) * num_latents
    return bits, max_bits, highest_prob


# TRAINING
def save_img(recon, label, path, idx, is_tiled, num_tiles):
    p = Path(path)
    p.mkdir(parents=True,exist_ok=True)
    print(f"recon image shape: {recon.shape}")
    filename = f"img{label}_{idx}.png"
    if is_tiled: 
        real_idx = idx // num_tiles
        split_num = idx % num_tiles
        filename = f"img{label}_{real_idx}_{split_num}.png"
    matplotlib.image.imsave(p / filename, recon.cpu().numpy(), cmap = "gray")	
    checkrecon = np.asarray(Image.open(p / filename).convert("L"))
    print(f"loaded image shape: {checkrecon.shape}")


# LAYERS RECONSTRUCTION
def recon_comparison(model, ds_test, names, descriptions, img_save_dir, is_tiled = False, num_tiles = 0):
    images = []
    for idx in range(len(ds_test)):
        (image, label) = ds_test[idx]    
        img = image.to(device).squeeze()
        images.append(img.cpu().numpy())

    #import ipdb; ipdb.set_trace()
    print("Original images to be reconstructed")
    output = np.hstack(images)
    #show_image(output)

    for layer, name, description in zip(model, names, descriptions):
        images = []
        for idx in range(len(ds_test)):
            (image, label) = ds_test[idx]    
            img = image.to(device).squeeze()
            
            for_recon = img.unsqueeze(0).unsqueeze(0)
            layer.eval()
            recon = layer.reconstruct(for_recon).squeeze()
            images.append(recon.cpu().numpy())

            recon_path = os.path.join(img_save_dir, f"data_recon_{names.index(name)}", f'{label}')
            orig_path = os.path.join(img_save_dir, 'data_original', f'{label}')
            jpg_path = os.path.join(img_save_dir, "data_jpg", f"{label}")
            save_img(recon, label, recon_path, idx, is_tiled, num_tiles)
            save_img(img, label, orig_path, idx, is_tiled, num_tiles)


            png_filename = f"img{label}_{idx}.png"
            if is_tiled:
                real_idx = idx // num_tiles
                split_num = idx % num_tiles
                png_filename = f"img{label}_{real_idx}_{split_num}.png"
            im = Image.open(os.path.join(orig_path, png_filename))
            #print("The size of the image before conversion :", end = "")
            #print(os.path.getsize(os.path.join(orig_path, f"img{label}_{idx}.png")))

            #converting to jpg
            gr_im = im.convert("L")

            #exporting the image
            p = Path(jpg_path)
            p.mkdir(parents = True, exist_ok=True)

            jpg_filename = f"img{label}_{idx}.jpeg"
            if is_tiled:
                real_idx = idx // num_tiles
                split_num = idx % num_tiles
                jpg_filename = f"img{label}_{real_idx}_{split_num}.jpeg"
            gr_im.save(os.path.join(jpg_path, jpg_filename))
            #print("The size of the image after conversion : ", end = "")
            #print(os.path.getsize(os.path.join(jpg_path, f"img{label}_{idx}.jpeg")))
    
        print(f"{name}: {description}")
        output = np.hstack(images)
        #show_image(output)


# HQA distortions in Fig 3
def get_rate_upper_bound(model, example_input):
    assert len(example_input.shape) == 4, "Expected (1, num_channels, x_h, x_w)"
    assert example_input.shape[0] == 1, "Please provide example with batch_size=1"
    
    z_e = model.encode(example_input)
    _, top_indices, _, _ = model.codebook(z_e)
        
    # assume worst case scenario: we have a uniform usage of all our codes
    rate_bound = top_indices[0].numel() * np.log2(model.codebook.codebook_slots)

    return rate_bound


def test(model, dl_test):
    model.eval()
    total_nll = []
    total_kl = []
    
    for x, _ in dl_test:
        img = x.to(device)       
        recon, orig, z_q, z_e, indices, kl, _ = model.forward_full_stack(img)       
        recon_loss = model[0].recon_loss(img, recon)        
        total_nll.append(recon_loss.item())
        if kl != 0:
            total_kl.append(kl.item())
        else:
            total_kl.append(kl)
    
    dims = np.prod(x.shape[1:])
    kl_mean = np.mean(total_kl)
    nll_mean = np.mean(total_nll)
    distortion_bpd = nll_mean / dims / np.log(2)
    rate_bpd = kl_mean / dims / np.log(2)
    elbo = -(nll_mean + kl_mean)
    
    rate_bound = get_rate_upper_bound(model, img[0].unsqueeze(0))
    
    return distortion_bpd, rate_bound


def get_rd_data(model, dl_test):
    dist = []
    rates = []
    
    for i, _ in enumerate(model):
        d, r = test(model[i], dl_test)
        dist.append(float(d))
        rates.append(float(r))
    
    return dist, rates


# Layer-wise interpolations
def interpolate(a, b, ds_test, vqvae, grid_x=16):
    images = []
    
    x_a,_ = ds_test[a]
    x_b,_ = ds_test[b]
    point_1 = vqvae.encode(x_a.unsqueeze(0).to(device))
    point_2 = vqvae.encode(x_b.unsqueeze(0).to(device))

    interpolate_x = np.linspace(point_1[0].cpu().numpy(), point_2[0].cpu().numpy(), grid_x)
    
    results = torch.Tensor(len(interpolate_x), 1, 32, 32)
    for i, z_e_interpolated in enumerate(interpolate_x):       
        z_e = torch.Tensor(z_e_interpolated).unsqueeze(0).to(device)
        z_q = vqvae.quantize(z_e)
        recon = vqvae.decode(z_q).squeeze() 
        results[i] = recon

    grid_img = make_grid(results.cpu(), nrow=grid_x)
    show_image(grid_img[0,:,:])


def show_original(idx, ds_test):
    x, _ = ds_test[idx]
    image = x.squeeze()
    show_image(image)
