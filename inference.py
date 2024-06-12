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

# Função para pré processamento do texto de entrada do modelo
def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
      text_norm = commons.intersperse(text_norm, 0)
  text_norm = torch.LongTensor(text_norm)
  return text_norm

# Carrefamos o json com as configurações do nosso modelo
#hps = utils.get_hparams_from_file("./configs/ljs_base.json")
hps = utils.get_hparams_from_file("./configs/base_0_speakers.json")

# Aqui, carregamos o checkpoint mais recente feito
net_g = SynthesizerTrn(
  len(symbols),
  hps.data.filter_length // 2 + 1,
  hps.train.segment_size // hps.data.hop_length,
  **hps.model).cuda()
_ = net_g.train()

_ = utils.load_checkpoint("logs/custom_model/G_7000.pth", net_g, None)

# Entrada da frase que vai ser enviada para o modelo de fala preocessar
# IMPORTANTE lembrar que o audio vai ser gerado de forma "diferente" sempre que for feito a inferencia, mesmo se tratando do mesmo checkpoint
stn_tst = get_text("Bilbo era muito rico e muito peculiar, e tinha sido a atração do Condado durante sessenta anos.", hps)
with torch.no_grad():
  x_tst = stn_tst.cuda().unsqueeze(0)
  x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
  audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

# Salva o audio em disco
ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

sf.write(f'audio.wav', audio, hps.data.sampling_rate)