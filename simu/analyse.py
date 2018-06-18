#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import synth

rate, data = scipy.io.wavfile.read('sample1.wav', )
assert(rate==44100)
assert(len(data.shape) == 1)

fft = np.fft.rfft(data, norm='ortho')
n = len(data)
sig = np.fft.irfft(fft, n, norm='ortho')
scipy.io.wavfile.write('sig.wav', rate, sig)
quit()

print(fft.shape)
print(fft.dtype)
plt.plot(np.abs(fft))

## Pick the 16 most significant frequencies
keep = 0
afft = np.abs(fft)
idx = np.argsort(afft)
idx = idx[-keep:]
print(idx)
print(fft[idx])
cfft = np.zeros(fft.shape, dtype=fft.dtype)
cfft[idx] = fft[idx]
plt.plot(np.flipud(np.abs(fft[idx])))
plt.plot(np.abs(cfft))

#sig = np.fft.irfft(cfft[idx], len(data))
sig = np.fft.irfft(fft, len(data))
scipy.io.wavfile.write('sig.wav', rate, sig)

plt.show()

