This is a modification of an existing implementation of the unconditional speech synthesizer in [DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS](https://arxiv.org/pdf/2009.09761.pdf).
This repo is a fork off of and builds from the [this implementation](https://github.com/philsyn/DiffWave-unconditional).

## Models
I train and provide two models -- both trained on the **Google Speech Commands SC09 dataset** (consisting of 1s, 16kHz spoken digits 'zero' through 'nine'). 
The two models are detailed:

| Model | Params | Training iters | Checkpoint | Hubconf name |
| -- | --: | :--: | :--: | :--: |
| DiffWave | 22M | 1M | [weights](https://github.com/RF5/DiffWave-unconditional/releases/download/v0.1/diffwave_sc09_1M_steps.pt), [train config](https://github.com/RF5/DiffWave-unconditional/releases/download/v0.1/config.json) | `diffwave_sc09` |
| DiffWave+Sashimi | 23.5M | 500k | [weights](https://github.com/RF5/DiffWave-unconditional/releases/download/v0.2/sashimi_sc09_500k_steps.pt), [tensorboard logs](https://tensorboard.dev/experiment/56xTVDTOSZileO67QXpBHw/#scalars) | `sashimi_diffwave_500k_sc09` |
| DiffWave+Sashimi | 23.5M | 800k | [weights](https://github.com/RF5/DiffWave-unconditional/releases/download/v0.2/sashimi_sc09_800k_steps.pt), [tensorboard logs](https://tensorboard.dev/experiment/56xTVDTOSZileO67QXpBHw/#scalars) | `sashimi_diffwave_800k_sc09` |

## Torchhub integration

This fork has been adapted from the original repo to provide torchhub integration for unconditional synthesis model, in addition to the diffwave+sashimi model. 
To use the model for inference, simply:

```python
model = torch.hub.load('RF5/DiffWave-unconditional', 'diffwave_sc09', device='cuda') # or device='cpu' if no cuda

audio = model.unconditional_generate(N=3) # number of samples you want to generate
# audio is now (N, 16000) 1s audio clips. You can save it as a wav or do whatever
# you like with it!
```

Other models available are `"sashimi_diffwave_500k_sc09"` and `"sashimi_diffwave_500k_sc09"`, which handles in exactly the same way and is the DiffWave model using the [SaShiMi](https://arxiv.org/abs/2202.09729) backbone adapted from the [original authors repo](https://github.com/HazyResearch/state-spaces/tree/main/sashimi).


## Training: 

- To continue training the model, run ```python distributed_train.py -c config.json```.

- To retrain the model, change the parameter ```ckpt_iter``` in the corresponding ```json``` file to ```-1``` and use the above command.

- To generate audio, run ```python inference.py -c config.json -n 16``` to generate 16 utterances. 

- Note, you may need to carefully adjust some parameters in the ```json``` file, such as ```data_path``` and ```batch_size_per_gpu```.

- The current config in the repo is for DiffWave+SaShiMi, [here is the train config of pure DiffWave](https://github.com/RF5/DiffWave-unconditional/releases/download/v0.1/config.json).


### SaShiMi DiffWave
I provide pretrained SaShiMi DiffWave models by adapting the [code provided by the authors](https://github.com/HazyResearch/state-spaces/tree/main/sashimi). This adapted SaShiMi+DiffWave model is defined in the `sashimi/` folder, and uses much code from the original author's repo -- thank you to them for making it public.

The original code appears to be very unstable unless it is used on a CUDA GPU with their custom CUDA kernel which a particular block in their model uses. 
I provide a copy of the CUDA kernel in the `extensions` folder -- this is duplicated from the [CUDA kernel in the original repo](https://github.com/HazyResearch/state-spaces/tree/main/extensions/cauchy) -- all credit to the original authors for providing it.

To install the CUDA kernel to use with SaShiMi, `cd` into the `extensions/cauchy` and install the module with `python setup.py install`. 

**NOTE**: To do this make sure you hae a gcc version newer than 4.9 but less than 9.0, otherwise nvcc or torch throws a fit. Without this, the S4 layer appears to use an exorbitant amount of memory for the number of parameters used. The official implementation of the S4 layer taken from the original repo even appears to slightly leak memory when not using these additional kernel or pykeops dependencies, so if you are training this model, I highly recommend installing the CUDA kernel.


## Acknowledgements

- Thank you to the author of the [original diffwave repo this repo is a fork from](https://github.com/philsyn/DiffWave-unconditional) (philsyn). This is the basis for the diffusion training pipeline.
- Thank you to the [DiffWave](https://arxiv.org/pdf/2009.09761.pdf) authors for developing the DiffWave model idea.
- Thank you to the [SaShiMi](https://arxiv.org/abs/2202.09729) authors for the paper and especially for providing a reference implementation of the autoregressive sashimi source code.