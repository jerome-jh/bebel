#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal

rate, data = scipy.io.wavfile.read('../snds/sample1.wav')
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
def nlargest(x, n):
    idx = np.argsort(x)
    return idx[-n:]

## Zero all the values in x but those with given indices
def zero_but(x, idx):
    y = np.zeros(x.shape, dtype=x.dtype)
    y[idx] = x[idx]
    return y

## Return indices of the n largest peaks
def nlargest_peak(x, n):
    idx = peak_detect(x)
    y = zero_but(x, idx)
    return nlargest(y, n)

## Return the indices of the peaks above given percentage of the max
def peak_above(x, p):
    m = np.amax(x)
    idx = peak_detect(x)
    peak = zero_but(x, idx)
    return np.where(peak > (m * p))[0]

## Generate signal
def gen(idx):
    print(idx, freq[idx], afft[idx])
    x = zero_but(afft, idx)
    plt.plot(freq, x)
    cfft = zero_but(fft, idx)
    return np.fft.irfft(cfft, len(data))

print(fft.shape)
print(fft.dtype)

freq = np.fft.rfftfreq(len(data), 1. / rate)
afft = np.abs(fft)
plt.plot(freq, afft)

keep = (8, 16, 32)

for k in keep:
    ## Generate signal keeping the largest peaks
    idx = nlargest_peak(afft, k)
    sig = gen(idx)
    scipy.io.wavfile.write('lp%d.wav'%k, rate, sig)

    ## Generate signal keeping the largest frequencies
    idx = nlargest(afft, k)
    sig = gen(idx)
    scipy.io.wavfile.write('lf%d.wav'%k, rate, sig)

pct = (0.15, 0.125, 0.10, 0.075)

for p in pct:
    ## Keep pure tones above level
    idx = peak_above(afft, p)
    sig = gen(idx)
    scipy.io.wavfile.write('pct%d.wav'%(100*p), rate, sig)

def hilbert_enveloppe(x):
    #env = np.abs(scipy.signal.hilbert(data, N=len(data) / 10))
    #env = np.abs(scipy.signal.hilbert(data, N=10*len(data)))
    ## Find fundamuntal freq
    #idx = nlargest_peak(afft, 1)[0]
    ## Filter LP at 1.5 fundamental
    #ffilt = 1.5 * freq[idx]
    #b, a = scipy.signal.butter(10, 2 * ffilt / rate, btype='lowpass')
    #x = scipy.signal.lfilter(b, a, data)
    #scipy.io.wavfile.write('filt.wav', rate, x)
    #print(ffilt)
    return np.abs(scipy.signal.hilbert(x))

def peak_enveloppe(x, plot=False):
    adata = np.abs(x)
    idx = peak_detect(adata)
    x = np.arange(len(x))
    y = adata[idx]
    i = np.interp(x, idx, y)
    if plot:
        plt.figure()
        plt.plot(idx, y)
        plt.plot(i)
        plt.plot(adata)
        plt.show()
    return i

## Apply lowpass filter to w, 0 < f < 1, where 1 is nyquist freq (rate / 2)
def lowpass(x, f):
    #b, a = scipy.signal.butter(10, f, btype='lowpass')
    #return scipy.signal.filtfilt(b, a, x)
    b = scipy.signal.firwin(200, f)
    a = 1.
    return scipy.signal.lfilter(b, a, x)

#sig = np.fft.irfft(cfft[idx], len(data))

plt.show()

plt.figure()

if True:
    env = peak_enveloppe(data)
else:
    env = hilbert_enveloppe(data)
scipy.io.wavfile.write('env.wav', rate, env)
plt.plot(env)

## Downsample the enveloppe
#x = scipy.signal.decimate(env, int(rate / erate))
erate = 210
decim = int(round(rate / erate))
#erate = int(round(rate / decim))
print(erate, decim)
fenv = lowpass(env, erate / rate)
scipy.io.wavfile.write('fenv.wav', rate, fenv)
plt.plot(fenv)

idx = decim * np.arange(int(len(data)/ decim))
xenv = fenv[idx]
scipy.io.wavfile.write('xenv.wav', erate, xenv)

plt.plot(idx, xenv)
plt.show()
