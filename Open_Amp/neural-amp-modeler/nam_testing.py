import os.path

import matplotlib

from nam.models.wavenet import WaveNet
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal.windows import chebwin

matplotlib.use('macosx')

filename = 'NAM/Marshall JCM 800 2203/JCM800 2203 - P5 B5 M5 T5 MV7 G10 - AZG - 700.nam'
filename = 'NAM/Mesa Boogie JP-2C Pack!/JP2C Capture 18 Ultimate Lead - Emil Rohbe.nam'

model_name = os.path.split(filename)[-1].split('.nam')[0]

with open(filename, 'r') as f:
    json_data = json.load(f)

if 'sample_rate' in json_data:
    fs = json_data['sample_rate']
else:
    fs = 48000


model = WaveNet(json_data['config']['layers'],
                json_data['config']['head'],
                json_data['config']['head_scale'], sample_rate=fs)
model.import_weights(json_data['weights'])

in_type = 'sine'
if in_type == 'sine':
    dur = 1.5
    gain = 10 ** (-20 / 20)
    f0 = 3951
    t_ax = torch.arange(0, int(dur * fs)) / fs
    x = gain * torch.sin(2 * torch.pi * f0 * t_ax)
else:
    fs_input, x = scipy.io.wavfile.read('/Users/alistaircarson/audio_datasets/dist_fx_192k/44k/test/input_32.wav')
    x = scipy.signal.resample(x, int(fs/fs_input * x.shape[-1]))
    x = torch.from_numpy(x)

#x = x[:fs]
with torch.no_grad():
    y = model(x).detach().cpu().numpy()

    if in_type == 'sine':
        y = y[-fs:]
        Y = np.fft.fft(y * chebwin(y.shape[0], at=-120))
        Y /= np.max(np.abs(Y))
        plt.plot(20 * np.log10(np.abs(Y)))
        plt.xlim([0, fs/2])
        plt.show()
    else:
        scipy.io.wavfile.write(model_name + '.wav', fs, y)

print(json_data)