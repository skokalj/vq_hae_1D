import os
from tqdm import tqdm
from glob import glob
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
from torchvision.transforms import Lambda, Compose

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from hqa_lightning1D_A import HQA as HQA_1Dim
import matplotlib.pyplot as plt

import librosa
import scipy.signal
import scipy.io.wavfile
from torch.utils.data import random_split


class AudioMNIST(Dataset):
    def __init__(self,
                 root_dir,
                 classes=range(10),
                 transform=None,
                 target_sample_rate=8000,
                 n_samples=16000,
                 preprocess_dataset=False):
        self.classes = classes
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.preprocess_dataset = preprocess_dataset
        if preprocess_dataset:
            self.spectrograms = self._load_spectrograms(root_dir)
        else:
            self.wav_files = self._load_wavs(root_dir)
        

    def __getitem__(self, index):
        if self.preprocess_dataset:
            return self.spectrograms[index]
        file_name, label = self.wav_files[index]
        spectogram = self._load_waveform_from_file(file_name)
        return spectogram, label
    
    def plot_waveform(self, index):
        file_name, label = self.wav_files[index]
        waveform, sample_rate = torchaudio.load(file_name)
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle("waveform")
        plt.show(block=False)
        figure.savefig(f'audio{label}.png')

    def _resize(self, signal):
        n_samples = signal.shape[1]
        if n_samples > self.n_samples:
            signal = signal[:, :self.n_samples]
        elif n_samples < self.n_samples:
            n_missing_samples = self.n_samples - n_samples
            signal = torch.nn.functional.pad(signal, (0, n_missing_samples))
        return signal

    def __len__(self):
        if self.preprocess_dataset:
            return len(self.spectrograms)
        return len(self.wav_files)

    def _load_wavs(self, root_dir):
        wav_files = list()
        cwd = os.getcwd()
        os.chdir(root_dir)
        for file_name in glob('**/*.wav', recursive=True):
            label = int(os.path.basename(file_name)[0])
            if label in self.classes:
                file_name = os.path.join(root_dir, file_name)
                wav_files.append((file_name, label))
        os.chdir(cwd)
        return wav_files

    def _load_spectrograms(self, root_dir):
        spectograms = list()
        cwd = os.getcwd()
        os.chdir(root_dir)
        print('Preprocessing dataset...')
        for file_name in tqdm(glob('**/*.wav', recursive=True)):
            label = int(os.path.basename(file_name)[0])
            if label in self.classes:
                waveform = self._load_waveform_from_file(file_name)
                if self.transform is not None:
                    spectograms.append((self.transform(waveform),label))
        os.chdir(cwd)
        return spectograms

    def _load_waveform_from_file(self,file_name):
        """Loads and preprocesses a waveform from a file

        Args:
            file_name (str): the name of the file to load from
        """
        waveform, sr = torchaudio.load(file_name)
        waveform = self._resample(waveform, sr)
        waveform = self._resize(waveform)
        if self.transform is not None:
            return self.transform(waveform)
        return waveform
        
    
    def _resample(self, signal, sample_rate):
        resampler = Resample(sample_rate, self.target_sample_rate)
        return resampler(signal)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    sample_rate = 16000
    EPOCHS=10
    '''
    transform = Compose([
        MelSpectrogram(sample_rate=sample_rate,
                       n_fft=1024,
                       n_mels=128,
                       hop_length=256,
                       normalized=False),
        AmplitudeToDB(top_db=np.inf),
        Lambda(lambda x: torch.unsqueeze(torch.flatten(x), 0))
    ])
    '''
    transform = Compose([
        Lambda(lambda x: torch.unsqueeze(torch.flatten(x), 0))
    ])
    ds = AudioMNIST('./AudioMNIST/data',
                    transform=transform,
                    target_sample_rate=8000,
                    n_samples=8000,
                    preprocess_dataset=False)
    ds.plot_waveform(0)
    print(ds[0][0].shape)
    #os._exit(0)
    ds_val, ds_train = random_split(ds, [len(ds) - int((3/4)*len(ds)),int((3/4)*len(ds))])

    train_dl = DataLoader(dataset=ds_train,
                    batch_size=64,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True)
    val_dl = DataLoader(dataset=ds_val,
                    batch_size=64,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True)
    torch.set_default_dtype(torch.float32)
    '''
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    CODEBOOK_SLOTS=256
    CODEBOOK_DIM=512
    for i in range(2):
        if i == 0:
            hqa = HQA_1Dim.init_bottom(
                input_feat_dim=1,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                codebook_slots=CODEBOOK_SLOTS,
                codebook_dim=CODEBOOK_DIM
            )
        else:
            hqa = HQA_1Dim.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                codebook_slots=CODEBOOK_SLOTS,
                codebook_dim=CODEBOOK_DIM
            )
        logger = TensorBoardLogger("tb_logs", name=f"HQA2o_AudioMNIST2_Layer{i}")
        trainer = Trainer(max_epochs=EPOCHS,
                          logger=logger,
                          strategy=DDPStrategy(find_unused_parameters=(i > 0))
                          )
        trainer.fit(hqa, train_dl, val_dl)
        hqa_prev = hqa




    model_save_path=os.path.join("tb_logs", f"HQA_Audio_2D_4096_MNIST_t6_{EPOCHS}.pt")    
    torch.save(hqa, model_save_path)  

    '''
    model_save_path=os.path.join("tb_logs", f"HQA_Audio_2D_4096_MNIST_t6_{EPOCHS}.pt")
    dl_test = val_dl
  
    hqa_model = torch.load(model_save_path)

    for i in range(2): 
        hqa=hqa_model[i]
        test_x, lab = next(iter(dl_test))
        hqa.eval()
        test_y = hqa.reconstruct(test_x)
        test_y = test_y.detach().cpu().numpy()
        batch_size= test_y.shape[0]
        figure1 = plt.figure()
        for k in range(batch_size):
            import ipdb; ipdb.set_trace()
            #griffinlim is when we train on spectrograms (2D) 
            #audio_signal = librosa.griffinlim(test_y[k])
            audio_signal = test_y[k]
            print(audio_signal, audio_signal.shape)
            num_channels, num_frames = audio_signal.shape
            time_axis = torch.arange(0, num_frames) / 8000
            plt.plot(time_axis, audio_signal.squeeze(), linewidth=1)            
# # write output
# scipy.io.wavfile.write('test.wav', fs, np.array(audio_signal, dtype=np.int16))
            plt.xticks([])
            plt.yticks([])
            plt.title(str(lab[k]))
        figure1.savefig(f'audiorecon{i}{k}.png')            
        figure2 = plt.figure(2)    
        test_x=test_x.detach().cpu().numpy()
        for k in range(batch_size):
            #griffinlim is when we train on spectrograms (2D) 
            #audio_signal = librosa.griffinlim(test_x[k])
            audio_signal = test_x[k]
            print(audio_signal, audio_signal.shape)
            num_channels, num_frames = audio_signal.shape
            time_axis = torch.arange(0, num_frames) / 8000
            plt.plot(time_axis, audio_signal.squeeze(), linewidth=1)   
            plt.xticks([])
            plt.yticks([])
            plt.title(str(lab[k]))
        figure2.savefig(f'audioorig{i}{k}.png')

    plt.close('all')



# audio_signal = librosa.core.spectrum.griffinlim(spectrogram)
# print(audio_signal, audio_signal.shape)

# # write output
    scipy.io.wavfile.write('test.wav', fs, np.array(audio_signal, dtype=np.int16))
