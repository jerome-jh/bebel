#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack

rate, data = scipy.io.wavfile.read('../snds/sample1.wav', )
assert(rate==44100)
assert(len(data.shape) == 1)

scale = 1. / 2**(16 - 1)
data = data * scale
fft = np.fft.rfft(data)
#fft = scipy.fftpack.rfft(data)
#n = len(data)
#sig = np.fft.irfft(fft, n)
#sig = scipy.fftpack.irfft(fft)
#print(fft.dtype)
#print(sig.dtype)
#sig = np.asarray(sig, dtype='f4')
#print(sig.dtype)
#scipy.io.wavfile.write('sig.wav', rate, sig)
#quit()

## Return the indices of peaks in x
def peak_detect(x):
    dx = np.ndarray((len(x) + 1,), dtype=x.dtype)
    dx[0] = x[1] - 0
    dx[1:-1] = x[1:] - x[0:-1]
    dx[-1] = 0 - x[-1]
    sign = np.where(dx >= 0, [1], [-1])
    pat = np.flipud(np.array([1, -1]))
    print(pat.dtype)
    print(sign.dtype)
    conv = np.convolve(sign, pat, mode='valid')
    maxi = np.where(conv == 2)[0]
    return maxi

## Return the indices of the n largest values in x
def keep_nlargest(x, n):
    idx = np.argsort(x)
    return idx[-n:]

## Zero all the values in x but those with given indices
def zero_but(x, idx):
    y = np.zeros(x.shape, dtype=x.dtype)
    y[idx] = x[idx]
    return y

print(fft.shape)
print(fft.dtype)

freq = np.fft.rfftfreq(len(data), 1. / rate)
afft = np.abs(fft)
plt.plot(freq, afft)

idx = peak_detect(afft)
afft = zero_but(afft, idx)
plt.plot(freq, afft)
print(idx.shape)

keep = 10000
idx = keep_nlargest(afft, keep)
afft = zero_but(afft, idx)
plt.plot(freq, afft)

print(idx, freq[idx], fft[idx])
cfft = zero_but(fft, idx)

#plt.plot(freq, np.flipud(np.abs(fft[idx])))
#plt.plot(freq, np.abs(cfft))

#sig = np.fft.irfft(cfft[idx], len(data))
sig = np.fft.irfft(cfft, len(data))
scipy.io.wavfile.write('sig.wav', rate, sig)

plt.show()

