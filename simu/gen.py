#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile

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

if __name__ == '__main__':
    par = open('param.txt', 'r')
    param = par.read()
    exec(param)

    ## Rate at which to gen wav
    rate = 48000
    data = np.zeros((int(round(rate * duration)),), dtype='f4')

    ## Generate sine table
    sine_step = 512
    t = np.arange(sine_step)
    sine = np.sin((2 * np.pi * t) / sine_step)

    assert(len(mag) == len(freq))
    t = np.arange(len(data))

    ## Synthetize sines
    ## TODO: proprer scaling
    for i in range(len(mag)):
        p = sine_step * freq[i] * t / rate
        p = np.asarray(np.rint(p), dtype='i4')
        p = p % sine_step
        data = data + mag[i] * sine[p] 
        
    scipy.io.wavfile.write('tone.wav', rate, data)

    ## Synthetize white noise abd replace data with it
    assert(len(noise_begin_t) == len(noise_end_t))
    noise_coord = [noise_begin_t, noise_end_t]
    for i in range(len(noise_coord)):
        noise_coord[i] = np.asarray(noise_coord[i], dtype='f4')
        noise_coord[i] = noise_coord[i] * rate
        noise_coord[i] = np.asarray(np.rint(noise_coord[i]), dtype='i4')
    for i in range(len(noise_coord[0])):
        b = noise_coord[0][i]
        e = noise_coord[1][i]
        n = aprng(e - b)
        data[b:e] = n

    ## Modulate with the enveloppe
    ## TODO: proprer scaling too
    te = np.arange(len(env_data)) * rate / env_rate
    ienv = np.interp(t, te, env_data)
    data = data * ienv

    scipy.io.wavfile.write('ring.wav', rate, data)

