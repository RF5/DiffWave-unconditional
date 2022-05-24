dependencies = ['torch', 'torchaudio', 'fastprogress', 'numpy', 'omegaconf']

import json
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from fastprogress.fastprogress import progress_bar
from torch import Tensor

from util import calc_diffusion_hyperparams
from WaveNet import WaveNet_Speech_Commands
from sashimi.model import SashimiDiffWave
from sashimi.config import DiffusionConfig
from omegaconf import OmegaConf


class DiffWaveWrapper(nn.Module):

    def __init__(self, diffwave: WaveNet_Speech_Commands, cfg: dict) -> None:
        super().__init__()
        self.diffwave = diffwave
        self.cfg = cfg
        self.z_dim = 16000 # z-dim is same as sequence length for diffusion model
        self.w_dim = 16000
        # dictionary of all diffusion hyperparameters
        self.diffusion_hyperparams   = calc_diffusion_hyperparams(**cfg["diffusion_config"])

    @torch.inference_mode()
    def unconditional_generate(self, N: int, progress=True, mb=None) -> Tensor:
        """ Generate `N` audio samples, returning a tensor of shape (N, 16000) 
    
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)
        Parameters:
        net (torch network):            the wavenet model
        size (tuple):                   size of tensor to be generated, 
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors 
        
        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """
        size = (N, 1, self.cfg['trainset_config']['segment_length'])
        _dh = self.diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert len(size) == 3
        assert size[-1] == 16000
        device = next(self.diffwave.parameters()).device

        x = torch.normal(0, 1, size=size).to(device)
        if progress: pb = progress_bar(range(T-1, -1, -1), parent=mb)
        else: pb = range(T-1, -1, -1)
        for t in pb:
            diffusion_steps = (t * torch.ones((size[0], 1))).to(device)  # use the corresponding reverse step
            epsilon_theta = self.diffwave((x, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size).to(device)  # add the variance term to x_{t-1}
        return x.squeeze(1)

    @torch.inference_mode()
    def generate_from_latent(self, z: Tensor, progress=True, mb=None) -> Tensor:
        """ Generate waveforms (N, 16000) from latent standard normal `z` (N, z_dim) """
        N = z.shape[0]
        size = (N, 1, self.cfg['trainset_config']['segment_length'])
        _dh = self.diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert len(size) == 3
        assert size[-1] == 16000
        device = next(self.diffwave.parameters()).device

        x = z[:, None].to(device) # (N, 1, 16000)
        if progress: pb = progress_bar(range(T-1, -1, -1), parent=mb)
        else: pb = range(T-1, -1, -1)
        for t in pb:
            diffusion_steps = (t * torch.ones((size[0], 1))).to(device)  # use the corresponding reverse step
            epsilon_theta = self.diffwave((x, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size).to(device)  # add the variance term to x_{t-1}
        return x.squeeze(1)

    @torch.inference_mode()
    def z2w(self, z: Tensor, progress=True, mb=None, interp_t=50) -> Tensor:
        """ Generate latent W vectors (N, w_dim) from latent standard normal `z` (N, z_dim).
        Latent vectors are defined as in Section 5.4 of the diffwave paper https://arxiv.org/pdf/2009.09761.pdf,
        namely using linear distances at t=50.
        """
        N = z.shape[0]
        size = (N, 1, self.cfg['trainset_config']['segment_length'])
        _dh = self.diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert len(size) == 3
        assert size[-1] == 16000
        device = next(self.diffwave.parameters()).device

        x = z[:, None].to(device) # (N, 1, 16000)
        if progress: pb = progress_bar(range(T-1, -1, -1), parent=mb)
        else: pb = range(T-1, -1, -1)
        for t in pb:

            diffusion_steps = (t * torch.ones((size[0], 1))).to(device)  # use the corresponding reverse step
            epsilon_theta = self.diffwave((x, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size).to(device)  # add the variance term to x_{t-1}

            if t == interp_t:
                # break, this is w
                w = x.squeeze(1) # (N, w_dim)
                break
        return w


    @torch.inference_mode()
    def generate_from_w(self, w: Tensor, progress=True, mb=None, interp_t=50) -> Tensor:
        """ Generate waveforms (N, 16000) from W latent space `w` (N, w_dim) """
        N = w.shape[0]
        size = (N, 1, self.cfg['trainset_config']['segment_length'])
        _dh = self.diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert len(size) == 3
        assert size[-1] == 16000
        device = next(self.diffwave.parameters()).device

        x = w[:, None].to(device) # (N, 1, 16000)
        if progress: pb = progress_bar(range(T-1, -1, -1), parent=mb)
        else: pb = range(T-1, -1, -1)
        for t in pb:
            # SKIP all t >= interp_t (50), i.e. only start when t <=49
            if t >= interp_t: continue

            diffusion_steps = (t * torch.ones((size[0], 1))).to(device)  # use the corresponding reverse step
            epsilon_theta = self.diffwave((x, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size).to(device)  # add the variance term to x_{t-1}
        return x.squeeze(1)

def diffwave_sc09(pretrained=True, progress=True, device='cuda'):
    """ DiffWave with WaveNet backbone: diffusion model trained on SC09 dataset. """
    with urllib.request.urlopen("https://github.com/RF5/DiffWave-unconditional/releases/download/v0.1/config.json") as url:
        config = json.loads(url.read().decode())

    gen_config              = config["gen_config"]
    wavenet_config          = config["wavenet_config"]      # to define wavenet
    diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    trainset_config         = config["trainset_config"]     # to read trainset configurations

    # predefine model
    model = WaveNet_Speech_Commands(**wavenet_config).to(device)
    
    if pretrained:
        # load checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/RF5/DiffWave-unconditional/releases/download/v0.1/diffwave_sc09_1M_steps.pt",
            progress=progress, map_location=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])

    model = DiffWaveWrapper(model, config)
    model = model.eval().to(device)

    print(f"[MODEL] DiffWave loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters")
    return model


def sashimi_diffwave_800k_sc09(pretrained=True, progress=True, device='cuda'):
    """ DiffWave with Sashimi backbone: diffusion model trained on SC09 dataset. """
    raise NotImplementedError()

    
def sashimi_diffwave_500k_sc09(pretrained=True, progress=True, device='cuda'):
    """ DiffWave with Sashimi backbone: diffusion model trained on SC09 dataset. """
    if 'cuda' not in str(device):
        raise NotImplementedError("SaShiMi is seemingly only accurately implement on CUDA GPUs with their custom CUDA kernel.")
    with urllib.request.urlopen("https://github.com/RF5/DiffWave-unconditional/releases/download/v0.1/config.json") as url:
        config = json.loads(url.read().decode())

    # predefine model
    model = SashimiDiffWave(OmegaConf.create(DiffusionConfig)).to(device)
    
    if pretrained:
        # load checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/RF5/DiffWave-unconditional/releases/download/v0.2/sashimi_sc09_500k_steps.pt",
            progress=progress, map_location=device
        )

        model.load_state_dict(checkpoint['model_state_dict'])

    model = DiffWaveWrapper(model, config)
    model = model.eval().to(device)
    print(f"[MODEL] SaShiMi DiffWave [500k] loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters")
    return model


def sashimi_diffwave_800k_sc09(pretrained=True, progress=True, device='cuda'):
    """ DiffWave with Sashimi backbone: diffusion model trained on SC09 dataset. """
    if 'cuda' not in str(device):
        raise NotImplementedError("SaShiMi is seemingly only accurately implement on CUDA GPUs with their custom CUDA kernel.")
    with urllib.request.urlopen("https://github.com/RF5/DiffWave-unconditional/releases/download/v0.1/config.json") as url:
        config = json.loads(url.read().decode())

    # predefine model
    model = SashimiDiffWave(OmegaConf.create(DiffusionConfig)).to(device)
    
    if pretrained:
        # load checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/RF5/DiffWave-unconditional/releases/download/v0.2/sashimi_sc09_800k_steps.pt",
            progress=progress, map_location=device
        )

        model.load_state_dict(checkpoint['model_state_dict'])

    model = DiffWaveWrapper(model, config)
    model = model.eval().to(device)
    print(f"[MODEL] SaShiMi DiffWave [800k] loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters")
    return model
