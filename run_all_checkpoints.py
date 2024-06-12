import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import pathlib
import soundfile as sf

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Local
import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
      text_norm = commons.intersperse(text_norm, 0)
  text_norm = torch.LongTensor(text_norm)
  return text_norm

hps = utils.get_hparams_from_file("./configs/base_0_speakers.json")

pathlib.Path('logs/custom_model/inferences/').mkdir(exist_ok=True, parents=True)
for model_name in os.listdir('logs/custom_model'):
  if model_name.startswith('G'):
    net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda()
    _ = net_g.train()

    _ = utils.load_checkpoint(f"logs/custom_model/{model_name}", net_g, None)

    stn_tst = get_text("Olá! Eu me chamo Programador Artificial. E está frase foi gerada usando um modelo de texto para fala!", hps)
    with torch.no_grad():
      x_tst = stn_tst.cuda().unsqueeze(0)
      x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
      audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    #ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

    sf.write(f'logs/custom_model/inferences/{model_name[:model_name.rfind(".")]}.wav', audio, hps.data.sampling_rate)