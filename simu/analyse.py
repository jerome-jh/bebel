#!/usr/bin/python3

##
## This script is for decomposing an audio sample in components, that can be used
## to synthetize an artificial version of the sample with very low CPU usage
## The components are:
## - tone: pure frequencies with largest amplitude are extracted
## - modulation: the enveloppe of the sample is extracted
## - percutions: regions of the signal with wideband noise are extracted
##
## For synthesis:
## - the pure frequencies are synthesized with zero phase
## - they are multiplied with the enveloppe
## - white noise is inserted or mixed in to recreation wideband signals
##
## Notes on the algos in this file:
## - tone extraction:
## we want to extract the frequencies with largest amplitude. However large peaks
## can be quite close, and keeping two close frequencies will cause beating during
## synthesis. It also does not enrich the sound to the hear that prefers harmonic
## frequencies.
## The algo finally retained is to do a first pass of peak detection on the FFT
## magnitude, linearly interpolate between the peaks, then do a second pass of
## peak detection, then take the nth largest. This will merge nearby peaks, while
## the largest peak of the group will keep its frequency and amplitude. Also lone
## peaks keep the same frequency and amplitude.
##
## - enveloppe extraction:
## we can use the Hilbert transform or a peak detection on the temporal signal.
## After low-pass filtering the result is exactly the same. The enveloppe is then
## downsampled to a few hundreds of Hertz
##
## - percusion noise detection:
## attempts have been made using the SNR of the tone but that was not reliable.
## Finally we take the first derivative of the amplitude and mix white noise where
## it peaks. That works very well for my samples but may not be always the case.
##
## Final remark: 
## The algo works well on the sample I tried it on. On other samples, it may
## require small or large adjustments.
##

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.signal

## Return the indices of peaks in x
## Can also return the valleys
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
    conv = np.convolve(sign, pat, mode='valid')
    maxi = np.where(conv == 2)[0]
    return maxi

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
def peak_above(x, p, plot=False):
    m = np.amax(x)
    idx = peak_detect(x)
    peak = zero_but(x, idx)
    idx = np.where(peak > (m * p))[0]
    if plot:
        plt.figure()
        plt.plot(x, label='x')
        plt.plot(peak, label='peak')
        plt.hlines(m, 0, len(x), label='max')
        plt.hlines(m * p, 0, len(x), label='threshold')
        plt.vlines(idx, np.amin(x), m, label='above')
        plt.legend()
        plt.title('peak_above')
    return idx

## Compute enveloppe with hilbert transform
def hilbert_enveloppe(x):
    return np.abs(scipy.signal.hilbert(x))

## Compute enveloppe using peaks
## Take the peaks then linear interpolate between them
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
def lowpass(x, f, order=200):
    #b, a = scipy.signal.butter(10, f, btype='lowpass')
    #return scipy.signal.filtfilt(b, a, x)
    b = scipy.signal.firwin(order, f)
    a = 1.
    #return scipy.signal.lfilter(b, a, x)
    return scipy.signal.filtfilt(b, a, x)

## From wikipedia https://en.wikipedia.org/wiki/Linear_congruential_generator
def lcg(modulus, a, c, seed):
    return (a * seed + c) % modulus

## Generate pseudo-random numbers given seed
prng_seed = 0x4A65726F
def prng():
    global prng_seed
    prng_seed = lcg(2**32, 1664525, 1013904223, prng_seed) 
    return prng_seed

## Return an array of n [-1, 1] pseudo random numbers
def aprng(n):
    a = np.ndarray((n,), dtype='u4')
    for i in range(n):
        a[i] = prng()
    return np.asarray(a / 2**31 - 1, dtype='f4')

## Return true if spectrum mean is higher than a percentage of the max
## Currently unused
def is_white_noise(spct, pct, t):
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
## Currently unused
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

## Normalize float array to [-1, 1]
def norm(a):
    return a / np.amax(a)

## Generate signal
def gen(idx, plot=False):
    x = zero_but(afft, idx)
    if plot:
        plt.figure()
        plt.plot(freq, x, label='tone spectrum')
        plt.legend()
    cfft = zero_but(fft, idx)
    return np.fft.irfft(cfft, len(data))

if __name__ == '__main__':
    ## Read input sample
    rate, data = scipy.io.wavfile.read('../snds/sample1.wav')
    assert(rate==44100)
    ## Want one channel wav
    assert(len(data.shape) == 1)
    duration = len(data) / rate

    ## File to output synthesis parameters
    par = open('param.txt', mode='w')
    print('duration =', duration, file=par)

    ## Fit input data to [-1, 1]
    scale = 1. / 2**(16 - 1)
    data = data * scale
    data = norm(data)
    ## Take the FFT
    fft = np.fft.rfft(data)
    freq = np.fft.rfftfreq(len(data), 1. / rate)
    afft = np.abs(fft)
    if False:
        ## LP filter to merge peaks that are close together
        freq_sep = 50 ## in Hz
        freq_cut = freq[1] / freq_sep
        safft = lowpass(afft, 2 * freq_cut, order=500)
    else:
        ## Take the peaks twice to merge those that are close together
        ## without affecting the lone ones
        safft = peak_enveloppe(peak_enveloppe(afft))

    ## Generate tone keeping the largest peaks
    #keep = (4, 8, 16, 32, 64)
    keep = (8,)
    for k in keep:
        idx = nlargest_peak(safft, k)
        ## Output a wav file with the inverse FFT of selected components
        sig = gen(idx)
        scipy.io.wavfile.write('lp%d.wav'%k, rate, sig)
        ## Output synthesis parameters
        print('mag =', tuple(2 * np.sqrt(2) * afft[idx] / len(data)), file=par)
        print('freq =', tuple(freq[idx]), file=par)
        if True:
            plt.figure()
            plt.plot(freq, afft, label='fft mag %d'%k)
            plt.plot(freq, safft, label='smoothed fft mag %d'%k)
            plt.vlines(freq[idx], 0, np.amax(afft))
            plt.legend()

    ## Take the enveloppe
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
    fenv = lowpass(env, erate / rate)
    ## Normalize back the enveloppe to still be full-scale
    ## This changes the relative levels compared to original data
    #fenv = norm(fenv)
    scipy.io.wavfile.write('fenv.wav', rate, fenv)
    
    xenv = decim * np.arange(int(len(data)/ decim))
    yenv = fenv[xenv]
    scipy.io.wavfile.write('yenv.wav', erate, yenv)

    print('env_rate =', erate, file=par)
    print('env_data =', tuple(yenv), file=par)
    
    plt.figure()
    plt.plot(env, label='enveloppe')
    plt.plot(fenv, label='enveloppe LP')
    plt.plot(xenv, yenv, label='enveloppe interp')
    plt.legend()

    ## Find where white noise should be inserted
    if False: ## Do not just flick that to True, both paths do not create the same vars
        twn, wn = locate_white_noise(data, rate)
        print(twn)
        print(wn)
    else:
        ## Take enveloppe first derivative
        dif = diff_pos(fenv)
        ## Look where peaks are high enough above mean
        pct = 15. / 100
        idx = peak_above(dif, pct)

    ## Generate white noise around points of detection
    white_noise_dur = 1e-3
    white_noise_samples = int(round(rate * white_noise_dur))
    wnb = np.zeros(data.shape, dtype='b')
    wnb[idx] = True
    hold = np.ones((white_noise_samples,), dtype='b')
    wnb = np.convolve(wnb, hold, mode='same') > 0
    noise = aprng(len(data))
    noise = wnb * noise 
    scipy.io.wavfile.write('noise.wav', rate, noise)

    ## Cap at zero and duration
    noise_beg_t = idx / rate - white_noise_dur
    if noise_beg_t[0] < 0:
        noise_beg_t[0] = 0
    noise_end_t = idx / rate + white_noise_dur
    if noise_end_t[-1] > duration:
        noise_end_t[-1] = duration
    print('noise_begin_t =', tuple(noise_beg_t), file=par)
    print('noise_end_t =', tuple(noise_end_t), file=par)

    ## Normalization factor to generate data full-scale
    ## TODO
    #print('norm =', 1 / np., file=par)

    plt.figure()
    plt.plot(fenv, label='enveloppe LP')
    plt.plot(dif, label='enveloppe dif')
    plt.hlines(pct * np.amax(dif), 0, len(dif), label='enveloppe dif')
    plt.vlines(idx, 0, np.amax([np.amax(dif), np.amax(fenv)]))
    plt.vlines(noise_beg_t * rate, 0, np.amax([np.amax(dif), np.amax(fenv)]))
    plt.vlines(noise_end_t * rate, 0, np.amax([np.amax(dif), np.amax(fenv)]))
    plt.legend()

    plt.figure()
    plt.plot(data, label='data')
    plt.plot(wnb, label='white noise detect')
    plt.plot(noise, label='white noise gen')
    plt.legend()

    par.close()

    plt.show()

