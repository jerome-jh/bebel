#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile

rate = 24000
nbits = 3
f = 440
a = 1
d = 2
t = np.arange(d * rate) / rate
w = 2 * np.pi * f
sig = a * np.sin(w*t)

scipy.io.wavfile.write('sig.wav', rate, sig / 2)

def bit(x, m):
    d = np.ndarray((m,), dtype='f4')
    v = int(round((x+1)*(m/2)))
    d[0:v] = 0.5
    d[v:m] = -0.5
    return d

n=2**nbits
pwn = np.zeros((d * rate * n), dtype='f4')
for i in range(t.shape[0]):
    x = sig[i]
    pwn[i*n:(i+1)*n] = bit(x,n)

scipy.io.wavfile.write('pwn.wav', n*rate, pwn)
