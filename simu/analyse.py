#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal

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
def peak_detect(x, valley=False):
    dx = np.ndarray((len(x) + 1,), dtype=x.dtype)
    dx[0] = x[1] - 0
    dx[1:-1] = x[1:] - x[0:-1]
    dx[-1] = 0 - x[-1]
    sign = np.where(dx >= 0, [1], [-1])
    if not valley:
        pat = np.flipud(np.array([1, -1]))
    else:
        pat = np.flipud(np.array([-1, 1]))
    print(pat.dtype)
    print(sign.dtype)
    conv = np.convolve(sign, pat, mode='valid')
    maxi = np.where(conv == 2)[0]
    return maxi

## Return true if spectrum mean is higher than a percentage of the max
def is_white_noise(spct, pct, t):
    #afft = np.abs(fft)
    #if False:
    #    afft = np.square(afft)
    if False:
        ## Take the square for energy
        nrj = np.square(spct)
        mean = np.mean(nrj, axis=0)
        maxi = np.amax(nrj, axis=0)
    else:
        mean = np.mean(spct, axis=0)
        maxi = np.amax(spct, axis=0)
    wn = mean > (pct * maxi / 100)
    if True:
        snr = maxi / mean
        plt.figure()
        plt.plot(snr, label='snr')
        plt.plot(mean, label='mean')
        plt.plot(maxi, label='maxi')
        plt.plot(wn * np.mean(maxi), label='iswn')
        plt.legend()
    if False:
        for i in range(10):
            print(t[i])
            plt.figure()
            plt.plot(spct[:,i])
            plt.hlines(mean[i], 0, spct.shape[0], label='mean')
            plt.hlines(maxi[i], 0, spct.shape[0], label='maxi')
            plt.legend()
    return wn

## Find regions where we can replace signal by white noise
def locate_white_noise(sig, rate):
    snr = 20
    wind = 128
    over = 2 * wind / 4
    f, t, s = scipy.signal.spectrogram(sig, fs=rate, nperseg=wind, noverlap=over, detrend=False, scaling='spectrum', mode='magnitude')
    print(s.shape)
    print(f.shape)
    print(t.shape)
    wn = is_white_noise(s, snr, t)
    return t, wn

## Compute differential x[n] - x[n-1], return same number of samples as x
def diff_pos(x):
    diff = np.ndarray(x.shape, dtype=x.dtype)
    diff[:-1] = x[1:] - x[:-1]
    diff[-1] = 0 - x[-1]
    return diff

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

## Compute enveloppe with hilbert transform
def hilbert_enveloppe(x):
    return np.abs(scipy.signal.hilbert(x))

# Compute enveloppe using peaks
def peak_enveloppe(x, plot=False):
    adata = np.abs(x)
    idx = peak_detect(adata)
    x = np.arange(len(x))
    y = adata[idx]
    i = np.interp(x, idx, y)
    if plot:
        plt.figure()
        plt.plot(adata, label='data')
        plt.plot(idx, y, label='peak')
        plt.plot(i, label='peak interp')
        plt.legend()
    return i

## Apply lowpass filter to w, 0 < f < 1, where 1 is nyquist freq (rate / 2)
def lowpass(x, f):
    #b, a = scipy.signal.butter(10, f, btype='lowpass')
    #return scipy.signal.filtfilt(b, a, x)
    b = scipy.signal.firwin(200, f)
    a = 1.
    return scipy.signal.lfilter(b, a, x)

## Generate signal
def gen(idx, plot=False):
    print(idx, freq[idx], afft[idx])
    x = zero_but(afft, idx)
    if plot:
        plt.figure()
        plt.plot(freq, x, label='timber spectrum')
        plt.legend()
    cfft = zero_but(fft, idx)
    return np.fft.irfft(cfft, len(data))

if __name__ == '__main__':
    rate, data = scipy.io.wavfile.read('../snds/sample1.wav')
    assert(rate==44100)
    assert(len(data.shape) == 1)
    
    scale = 1. / 2**(16 - 1)
    data = data * scale
    fft = np.fft.rfft(data)
    print(fft.shape)
    print(fft.dtype)

    freq = np.fft.rfftfreq(len(data), 1. / rate)
    afft = np.abs(fft)
    
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
    
    #sig = np.fft.irfft(cfft[idx], len(data))
    
    if True:
        env = peak_enveloppe(data)
    else:
        env = hilbert_enveloppe(data)
    scipy.io.wavfile.write('env.wav', rate, env)
    
    ## Downsample the enveloppe
    #x = scipy.signal.decimate(env, int(rate / erate))
    erate = 210
    decim = int(round(rate / erate))
    #erate = int(round(rate / decim))
    print(erate, decim)
    fenv = lowpass(env, erate / rate)
    scipy.io.wavfile.write('fenv.wav', rate, fenv)
    
    idx = decim * np.arange(int(len(data)/ decim))
    xenv = fenv[idx]
    scipy.io.wavfile.write('xenv.wav', erate, xenv)
    
    plt.figure()
    plt.plot(freq, afft, label='fft mag')
    plt.legend()

    plt.figure()
    plt.plot(env, label='enveloppe')
    plt.plot(fenv, label='enveloppe LP')
    plt.plot(idx, xenv, label='enveloppe interp')
    plt.legend()

    twn, wn = locate_white_noise(data, rate)
    print(twn)
    print(wn)

    plt.show()

