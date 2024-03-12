import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import RelaxedOneHotCategorical, Categorical
from torch.optim.lr_scheduler import _LRScheduler


import lightning.pytorch as pl
from sklearn.manifold import TSNE


def mish(x):
    return x * torch.tanh(F.softplus(x))

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)
    
class FlatCA(_LRScheduler):
    def __init__(self, optimizer, steps, eta_min=0, last_epoch=-1):
        self.steps = steps
        self.eta_min = eta_min
        super(FlatCA, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_list = []
        T_max = self.steps / 3
        for base_lr in self.base_lrs:
            # flat if first 2/3
            if 0 <= self._step_count < 2 * T_max:
                lr_list.append(base_lr)
            # annealed if last 1/3
            else:
                lr_list.append(
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * (self._step_count - 2 * T_max) / T_max))
                    / 2
                )
            return lr_list

class Encoder(nn.Module):
    """ Downsamples by a fac of 2 """

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0, batch_norm=1):
        super().__init__()
        blocks = [
            nn.Conv1d(in_feat_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            Mish(),
        ]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))
            if batch_norm == 2:
                blocks.append(nn.BatchNorm1d(hidden_dim))

        blocks.append(nn.Conv1d(hidden_dim, codebook_dim, kernel_size=1))
        if(batch_norm):
            blocks.append(nn.BatchNorm1d(codebook_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)

class Encoder2(nn.Module):
    """ Downsamples by a fac of 4 """
    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0,batch_norm=1):
        super().__init__()
        blocks = [
            nn.Conv1d(in_feat_dim, hidden_dim // 2, kernel_size=7, stride=4, padding=2),
            Mish(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            Mish(),
        ]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))
            if batch_norm==2:
                blocks.append(nn.BatchNorm1d(hidden_dim))
        blocks.append(nn.Conv1d(hidden_dim, codebook_dim, kernel_size=1))

        if(batch_norm):
            blocks.append(nn.BatchNorm1d(codebook_dim))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)



class Decoder(nn.Module):
    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim # num channels on bottom layer
        blocks = [nn.Conv1d(in_feat_dim, hidden_dim, kernel_size=3, padding=1), Mish()]
        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))

        blocks.extend([
                Upsample(),
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                Mish(),
                nn.Conv1d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
        ])
 
        if very_bottom is True:
            blocks.append(nn.Tanh())       
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = x.float()
        return self.blocks(x)

class Decoder2(nn.Module):
    """ Upsamples by a fac of 2 """
    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim # num channels on bottom layer
        
        blocks = [nn.Conv1d(in_feat_dim, hidden_dim, kernel_size=3, padding=1), Mish()]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))

        blocks.extend([
                Upsample(),
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                Mish(),
                nn.Conv1d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
                Upsample(),
        ])

        if very_bottom is True:
            blocks.append(nn.Tanh()) 
            
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = x.float()
        return self.blocks(x)



class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channel, channel, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(channel, in_channel, kernel_size=3, padding=1)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = mish(x)
        x = self.conv_2(x)
        x = x + inp
        return mish(x)

class VQCodebook(nn.Module):
    def __init__(self, codebook_slots, codebook_dim, temperature=0.5):
        super().__init__()
        self.codebook_slots = codebook_slots
        self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.codebook = nn.Embedding(codebook_slots, codebook_dim)
        self.log_slots_const = np.log(self.codebook_slots)

    def z_e_to_z_q(self, z_e, soft=True):
        bs, feat_dim, w = z_e.shape
        assert feat_dim == self.codebook_dim
        z_e = z_e.permute(0, 2, 1).contiguous()
        z_e_flat = z_e.view(bs * w, feat_dim)
        codebook = self.codebook.weight
        codebook_sqr = torch.sum(codebook ** 2, dim=1)
        z_e_flat_sqr = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)

        distances = torch.addmm(
            codebook_sqr + z_e_flat_sqr, z_e_flat, codebook.t(), alpha=-2.0, beta=1.0
        )

        if soft is True:
            dist = RelaxedOneHotCategorical(self.temperature, logits=-distances)
            soft_onehot = dist.rsample()
            hard_indices = torch.argmax(soft_onehot, dim=1).view(bs, w)
            z_q = (soft_onehot @ codebook).view(bs, w, feat_dim)
            
            # entropy loss
            KL = dist.probs * (dist.probs.add(1e-9).log() + self.log_slots_const)
            KL = KL.view(bs, w, self.codebook_slots).sum(dim=(1,2)).mean()
            
            # probability-weighted commitment loss    
            commit_loss = (dist.probs.view(bs, w, self.codebook_slots) * distances.view(bs, w, self.codebook_slots)).sum(dim=(1,2)).mean()
        else:
            with torch.no_grad():
                dist = Categorical(logits=-distances)
                hard_indices = dist.sample().view(bs, w)
                hard_onehot = (
                    F.one_hot(hard_indices, num_classes=self.codebook_slots)
                    .type_as(codebook)
                    .view(bs * w , self.codebook_slots)
                )
                z_q = (hard_onehot @ codebook).view(bs, w, feat_dim)
                
                # entropy loss
                KL = dist.probs * (dist.probs.add(1e-9).log() + np.log(self.codebook_slots))
                KL = KL.view(bs, w, self.codebook_slots).sum(dim=(1,2)).mean()

                commit_loss = 0.0

        z_q = z_q.permute(0, 2, 1)

        return z_q, hard_indices, KL, commit_loss

    def lookup(self, ids: torch.Tensor):
        #return F.embedding(ids, self.codebook).permute(0, 3, 1, 2)
        codebook = self.codebook.weight
        return F.embedding(ids, codebook).permute(0, 2, 1)

    def quantize(self, z_e, soft=False):
        with torch.no_grad():
            z_q, indices, _, _ = self.z_e_to_z_q(z_e, soft=soft)
        return z_q, indices

    def quantize_indices(self, z_e, soft=False):
        with torch.no_grad():
            _, indices, _, _ = self.z_e_to_z_q(z_e, soft=soft)
        return indices

    def forward(self, z_e, soft=True):
        z_q, indices, kl, commit_loss = self.z_e_to_z_q(z_e, soft)
        return z_q, indices, kl, commit_loss

class GlobalNormalization1(torch.nn.Module):

    def __init__(self, feature_dim, scale=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("running_ave", torch.zeros(1, self.feature_dim, 1))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        if self.scale:
            self.register_buffer("running_sq_diff", torch.zeros(1, self.feature_dim, 1))

    def forward(self, inputs):

        if self.training:
            # Update running estimates of statistics
            frames_in_input = inputs.shape[0] * inputs.shape[2]
            updated_running_ave = (
                self.running_ave * self.total_frames_seen + inputs.sum(dim=(0, 2), keepdim=True)
            ) / (self.total_frames_seen + frames_in_input)

            if self.scale:
                # Update the sum of the squared differences between inputs and mean
                self.running_sq_diff = self.running_sq_diff + (
                    (inputs - self.running_ave) * (inputs - updated_running_ave)
                ).sum(dim=(0, 2), keepdim=True)

            self.running_ave = updated_running_ave
            self.total_frames_seen = self.total_frames_seen + frames_in_input
        else:
            return inputs

        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = (inputs - self.running_ave) / std
        else:
            inputs = inputs - self.running_ave

        return inputs

    def unnorm(self, inputs):
        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = inputs*std + self.running_ave
        else:
            inputs = inputs + self.running_ave

        return inputs
    


class HQA(pl.LightningModule):
    VISUALIZATION_DIR = 'vis'
    SUBDIRS=[VISUALIZATION_DIR]
    def __init__(
        self,
        input_feat_dim,
        prev_model=None,
        codebook_slots=256,
        codebook_dim=256,
        enc_hidden_dim=16,
        dec_hidden_dim=32,
        gs_temp=0.667,
        num_res_blocks=0,
        lr=4e-4,
        decay=True,
        clip_grads=False,
        codebook_init='normal',
        output_dir = "CodeCos2R",
        layer = 0 ,
        KL_coeff = 0.2,
        CL_coeff = 0.001,
        Cos_coeff = 0.7,
        batch_norm = 1,
        reset_choice = 0,
        cos_reset = 1,
        compress = 2

    ):
        super().__init__()
        self.save_hyperparameters(ignore=['prev_model'])
        self.prev_model = prev_model
        self.codebook = VQCodebook(codebook_slots, codebook_dim, gs_temp)
        if compress == 2:
            self.encoder = Encoder(input_feat_dim, codebook_dim, enc_hidden_dim, num_res_blocks=num_res_blocks,batch_norm=True)
            self.decoder = Decoder(
                codebook_dim,
                input_feat_dim,
                dec_hidden_dim,
                very_bottom=prev_model is None,
                num_res_blocks=num_res_blocks
            )
        else:
            self.encoder = Encoder2(input_feat_dim, codebook_dim, enc_hidden_dim, num_res_blocks=num_res_blocks,batch_norm=True)
            self.decoder = Decoder2(
                codebook_dim,
                input_feat_dim,
                dec_hidden_dim,
                very_bottom=prev_model is None,
                num_res_blocks=num_res_blocks
            )
        self.normalize = GlobalNormalization1(codebook_dim, scale=True)
        self.out_feat_dim = input_feat_dim
        self.codebook_dim = codebook_dim
        self.lr = lr
        self.decay = decay
        self.clip_grads = clip_grads
        self.layer = layer
        self.KL_coeff = torch.tensor(KL_coeff)
        self.CL_coeff = torch.tensor(CL_coeff)
        self.Cos_coeff = torch.tensor(Cos_coeff)
        self.reset_choice = reset_choice
        self.cos_reset = cos_reset
        
        torch.set_default_dtype(torch.float32)
        
        # Tells pytorch lightinig to use our custom training loop
        self.automatic_optimization = False
        
        self.init_codebook(codebook_init)
        self.create_output = output_dir is not None 
        if self.create_output:
            self.output_dir = output_dir
            try:
                os.mkdir(output_dir)
                for subdir in HQA.SUBDIRS:
                    path = f'{output_dir}/{subdir}'
                    os.mkdir(path)
                    print(path)
                    os.mkdir(f'{path}/layer{len(self)}')
            except OSError:
                pass    
    
    @torch.no_grad()
    def init_codebook(self, codebook_init):
        if codebook_init == 'uniform':
            self.codebook.codebook.weight.data.uniform_(-1./self.codebook.codebook_slots, 1./self.codebook.codebook_dim)
            print("Uniform Codebook initialization")
        elif codebook_init == 'normal':
            self.codebook.codebook.weight.data.normal_()
            print("Normal Codebook initialization")
        else:
            raise Exception("Invalid codebook initialization")
            
            
    def forward(self, x, soft = True):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_q, indices, kl, commit_loss = self.codebook(z_e, soft)
        z_e_lower_tilde = self.decoder(z_q)
        return z_e_lower_tilde, z_e_lower, z_q, z_e, indices, kl, commit_loss
    
    def on_train_start(self):
        self.code_count = torch.zeros(self.codebook.codebook_slots, device=self.device, dtype=torch.float64)
        self.codebook_resets = 0
        if(self.cos_reset):
            self.Cos_coeff = self.Cos_coeff*int(self.layer == 0)
        
    
    def cos_loss(self,x):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_q, indices, kl, commit_loss = self.codebook(z_e)
        z_e_lower_tilde = self.decoder(z_q)
        cos_loss=torch.max(1-F.cosine_similarity(z_e_lower, z_e_lower_tilde, dim = 1),torch.zeros(z_e_lower.shape[0], z_e_lower.shape[2], device=self.device)).sum(dim=1).mean()
        return cos_loss

    def val_cos_loss(self,x):
        z_e_lower_tilde, z_e_lower, z_q, z_e, indices, kl, commit_los =self(x, soft=False)
        cos_loss=torch.max(1-F.cosine_similarity(z_e_lower, z_e_lower_tilde, dim = 1),torch.zeros(z_e_lower.shape[0], z_e_lower.shape[2], device=self.device)).sum(dim=1).mean()
        return cos_loss


    def get_training_loss(self, x):
        recon, recon_test, lll, _, indices, KL, commit_loss = self(x)
        #import ipdb; ipdb.set_trace()
        recon_loss = self.recon_loss(self.encode_lower(x), recon)
        cos_loss = self.cos_loss(x)
        dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
        #loss = recon_loss/dims + 0.001*KL/dims + 0.001*(commit_loss)/dims
        loss = self.Cos_coeff*cos_loss/dims + recon_loss/dims + self.KL_coeff*KL/dims + self.CL_coeff*(commit_loss)/dims
        return cos_loss, recon_loss, loss, indices, KL, commit_loss
    
    def get_validation_loss(self, x):
        recon, recon_test, _, _, indices, KL, commit_loss = self(x, soft=False)
        recon_loss = self.recon_loss(self.encode_lower(x), recon)
        val_cos_loss = self.val_cos_loss(x)
        dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
        loss = self.Cos_coeff*val_cos_loss/dims + recon_loss/dims + self.KL_coeff*KL/dims + self.CL_coeff*(commit_loss)/dims
        return val_cos_loss, recon_loss, loss, indices, KL, commit_loss    

    def recon_loss(self, orig, recon):
        return F.mse_loss(orig, recon, reduction='none').sum(dim=(1,2)).mean()
    
    def decay_temp_linear(self, step, total_steps, temp_base, temp_min=0.001):
        factor = 1.0 - (step/total_steps)
        return temp_min + (temp_base - temp_min) * factor

    def training_step(self,batch, batch_idx):
        x,_ = batch
        
        # anneal temperature
        if self.decay:
                self.codebook.temperature = self.decay_temp_linear(step = self.global_step+1, 
                                                                   total_steps = self.trainer.max_epochs * self.trainer.num_training_batches, 
                                                                   temp_base= self.codebook.temperature)
        
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
    
        cos_loss, recon_loss, loss, indices, kl_loss, commit_loss = self.get_training_loss(x.float())
    
        optimizer.zero_grad()
        
        self.manual_backward(loss)
        
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        indices_onehot = F.one_hot(indices, num_classes=self.codebook.codebook_slots).float()
        self.code_count = self.code_count + indices_onehot.sum(dim=(0, 1))

        
        if batch_idx > 0 and batch_idx % 15 == 0:
            self.reset_least_used_codeword()
            if self.create_output and self.codebook_resets % 30 == 0:
                tsne = self.visualize_codebook()
                df = pd.DataFrame(tsne,
                    columns=['tsne-2d-one', 'tsne-2d-two'])
                [i for i in range(len(df))]

                plt.figure(figsize=(16,10))
                scplot=sns.scatterplot(
                    x="tsne-2d-one", y="tsne-2d-two",
                    #hue=y,
                    color='black',
                    #palette=sns.color_palette("hls", 10),
                    data=df,
                    legend=False,
                    #alpha=0.3,
                    #s=20
                )

                if ( not os.path.exists(f'{self.output_dir}/{HQA.VISUALIZATION_DIR}/layer{len(self)}')):
                    os.makedirs(f'{self.output_dir}/{HQA.VISUALIZATION_DIR}/layer{len(self)}')
                fig = scplot.get_figure()
                fig.savefig(f'{self.output_dir}/{HQA.VISUALIZATION_DIR}/layer{len(self)}/reset{self.codebook_resets}.png')
                plt.close()

        self.log("loss", loss, prog_bar=True)
        self.log("cos_loss", cos_loss, prog_bar=True)
        self.log("recon", recon_loss, prog_bar=True)
        self.log("kl", kl_loss, prog_bar=True)
        self.log("commit", commit_loss, prog_bar=True)
        return loss


    def visualize_codebook(self):
        """ Perform t-SNE visualization on the VQ-Codebook """
        latents = self.codebook.codebook.weight.data.detach().cpu().numpy()
        tsne = TSNE(n_components=2)
        latents_tsne = tsne.fit_transform(latents)
        return latents_tsne

    @torch.no_grad()
    def reset_least_used_codeword(self):
        max_count, most_used_code = torch.max(self.code_count, dim=0)
        frac_usage = self.code_count / max_count
        z_q_most_used = self.codebook.lookup(most_used_code.view(1, 1)).squeeze()
        min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
        reset_factor = 100
        if self.reset_choice == 0:
            reset_factor = 100
        else:
            reset_factor = (1*(self.codebook_resets+1))
        if min_frac_usage < 0.03:
            #print(f'reset code {min_used_code}')
            moved_code = z_q_most_used + torch.randn_like(z_q_most_used) / reset_factor
            self.codebook.codebook.weight.data[min_used_code] = moved_code
        self.code_count = torch.zeros_like(self.code_count, device=self.device)
        self.codebook_resets += 1    
    

    def validation_step(self, val_batch, batch_idx):
        x,_ = val_batch
        cos_loss, recon_loss, loss, indices, kl_loss, commit_loss = self.get_validation_loss(x)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        self.log("val_cos_loss", cos_loss, prog_bar=False,sync_dist=True)
        self.log("val_recon", recon_loss, prog_bar=False, sync_dist=True)
        self.log("val_kl", kl_loss, prog_bar=False,  sync_dist=True)
        self.log("val_commit", commit_loss, prog_bar=False,sync_dist=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x,_ = test_batch
        cos_loss, recon_loss, loss, indices, kl_loss, commit_loss = self.get_validation_loss(x)
        self.log("tst_loss", loss, prog_bar=False)
        self.log("tst_cos_loss", cos_loss, prog_bar=False)
        self.log("tst_recon", recon_loss, prog_bar=False)
        self.log("tst_kl", kl_loss, prog_bar=False)
        self.log("tst_commit", commit_loss, prog_bar=False)
        return loss    

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        lr_scheduler = FlatCA(optimizer, steps=1, eta_min=4e-5)
        return [optimizer], [lr_scheduler]
    
    def encode_lower(self, x):
        if self.prev_model is None:
            return x
        else:
            with torch.no_grad():
                z_e_lower = self.prev_model.encode(x)
                z_e_lower = self.normalize(z_e_lower)
            return z_e_lower
    def encode(self, x):
        with torch.no_grad():
            z_e_lower = self.encode_lower(x)
            z_e = self.encoder(z_e_lower)
        return z_e
        
    def decode_lower(self, z_q_lower):
        with torch.no_grad():
            recon = self.prev_model.decode(z_q_lower)           
        return recon

    def decode(self, z_q):
        with torch.no_grad():
            if self.prev_model is not None:
                z_e_u = self.normalize.unnorm(self.decoder(z_q))
                z_q_lower_tilde = self.prev_model.quantize(z_e_u)
                recon = self.decode_lower(z_q_lower_tilde)
            else:
                recon = self.decoder(z_q)
        return recon

    def quantize(self, z_e):
        z_q, _ = self.codebook.quantize(z_e)
        return z_q
    
    def reconstruct_average(self, x, num_samples=10):
        """Average over stochastic edecodes"""
        b, c, h = x.shape
        result = torch.empty((num_samples, b, c, h))#.to(device)

        for i in range(num_samples):
            result[i] = self.decode(self.quantize(self.encode(x)))
        return result.mean(0)

    def reconstruct(self, x):
        return self.decode(self.quantize(self.encode(x)))
    
    def reconstruct_from_codes(self, codes):
        return self.decode(self.codebook.lookup(codes))
    
    def reconstruct_from_z_e(self, z_e):
        return self.decode(self.quantize(z_e))
    
    def __len__(self):
        i = 1
        layer = self
        while layer.prev_model is not None:
            i += 1
            layer = layer.prev_model
        return i

    def __getitem__(self, idx):
        max_layer = len(self) - 1
        if idx > max_layer:
            print(idx[0])
            raise IndexError("layer does not exist")

        layer = self
        for _ in range(max_layer - idx):
            layer = layer.prev_model
        return layer

    def parameters(self, prefix="", recurse=True):
        for module in [self.encoder, self.codebook, self.decoder]:
            for name, param in module.named_parameters(recurse=recurse):
                yield param
    
    @classmethod
    def init_higher(cls, prev_model, **kwargs):
        model = HQA(prev_model.codebook.codebook_dim, prev_model=prev_model, **kwargs)
        model.prev_model.eval()
        return model
    
    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        model = HQA(input_feat_dim,prev_model=None, **kwargs)
        return model

