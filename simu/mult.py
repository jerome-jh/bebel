#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal

## Multiply enveloppe and ifft generated signal

rate, env = scipy.io.wavfile.read('xenv.wav')
print(rate)
assert(rate==210)
assert(len(env.shape) == 1)

decim = int(round(44100 / rate))
## Interpolate the enveloppe
x = np.arange((len(env) + 1) * decim)
idx = np.arange(len(env)) * decim
ienv = np.interp(x, idx, env)
plt.plot(ienv)
plt.plot(idx, env)

def norm(x):
    return x / np.amax(x)

def mult(ifile, ofile, env):
    rate, data = scipy.io.wavfile.read(ifile)
    assert(rate==44100)
    assert(len(data.shape) == 1)
    # crop enveloppe
    e = env[:len(data)]
    x = e * data
    scipy.io.wavfile.write(ofile, rate, norm(x))

mult('lf2.wav', 'elf2.wav', ienv)
mult('lf4.wav', 'elf4.wav', ienv)
mult('lp2.wav', 'elp2.wav', ienv)
mult('lp4.wav', 'elp4.wav', ienv)
mult('med.wav', 'emed.wav', ienv)

plt.show()
