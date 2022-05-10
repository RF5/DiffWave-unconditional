This is a reimplementaion of the unconditional waveform synthesizer in [DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS](https://arxiv.org/pdf/2009.09761.pdf).

## Usage: 

- To continue training the model, run ```python distributed_train.py -c config.json```.

- To retrain the model, change the parameter ```ckpt_iter``` in the corresponding ```json``` file to ```-1``` and use the above command.

- To generate audio, run ```python inference.py -c config.json -n 16``` to generate 16 utterances. 

- Note, you may need to carefully adjust some parameters in the ```json``` file, such as ```data_path``` and ```batch_size_per_gpu```.

## Pretrained models and generated samples:
- [model](https://github.com/philsyn/DiffWave-unconditional/tree/master/exp/ch256_T200_betaT0.02/logs/checkpoint)
- [samples](https://github.com/philsyn/DiffWave-unconditional/tree/master/exp/ch256_T200_betaT0.02/speeches)

## Torchhub integration

This fork has been adapted from the original repo to provide torchhub integration for unconditional synthesis on the original model. 
To use the model for inference, simply:

```python
model = torch.hub.load('RF5/DiffWave-unconditional', 'diffwave_sc09', device='cuda') # or device='cpu' if no cuda

audio = model.unconditional_generate(N=3) # number of samples you want to generate
# audio is now (N, 16000) 1s audio clips. You can save it as a wav or do whatever
# you like with it!
```

