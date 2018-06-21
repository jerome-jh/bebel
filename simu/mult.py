#!/usr/bin/python3

import glob
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal

## Multiply enveloppe and ifft generated signal

rate, env = scipy.io.wavfile.read('yenv.wav')
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

def mult(ifile, env):
    rate, data = scipy.io.wavfile.read(ifile)
    assert(rate==44100)
    assert(len(data.shape) == 1)
    # crop enveloppe
    e = env[:len(data)]
    x = e * data
    return rate, norm(x)

def mix(x, y):
    return (x + y) / 2

def rep(x, y):
    return np.where(y != 0, y, x)

## Shape the noise with enveloppe
#nrate, noise = scipy.io.wavfile.read('noise.wav')
nrate, noise = mult('noise.wav', ienv)
gl = ('lf*.wav', 'lp*.wav', 'pct*.wav')
for g in gl:
    for f in glob.glob(g):
        rate, sig = mult(f, ienv)
        assert(rate == nrate)
        sig = rep(sig, noise)
        scipy.io.wavfile.write('e' + f, rate, norm(sig))

rate, data = scipy.io.wavfile.read('../snds/sample1.wav')
assert(rate==44100)
assert(len(data.shape) == 1)
scipy.io.wavfile.write('inf.wav', rate, norm(data))

#plt.show()
