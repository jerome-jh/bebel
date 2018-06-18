#!/usr/bin/python3

import numpy as np
import scipy.io.wavfile

coef = np.array((
  ( 5, 1),
  (1500, 1),
  (1600, 1),
  (3000, 1.2),
  (3200, 1.2),
  (6000, 0.75),
  (6400, 0.75)
))

rate = 44100

def synth(freq, ampl, duration):
    print(freq)
    print(ampl)
    freq = np.asarray(freq)
    ampl = np.asarray(ampl)
    assert(freq.shape == ampl.shape)
    w = 2 * np.pi * freq
    ## Normalize amplitude
    # a.append(c[1] / len(coef))
    t = np.arange(int(round(duration * rate)), dtype='f4') / rate
    d = np.zeros(t.shape, dtype='f4')
    for i in range(w.shape[0]):
        d += ampl[i] * np.sin(w[i] * t)
    print(t)
    return d

if __name__ == '__main__':
    data = synth([440], [1], 2)
    scipy.io.wavfile.write('sin.wav', rate, data)

    data = synth(coef[:,0], coef[:,1] / coef.shape[0], 2)
    scipy.io.wavfile.write('data.wav', rate, data)

