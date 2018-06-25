#!/usr/bin/python3

import numpy as np

if __name__ == '__main__':
    par = open('param.txt', 'r')
    param = par.read()
    exec(param)

    ## Rate at which to gen wav
    rate = 48000
    data = np.ndarray((int(round(rate * duration)),), dtype='f4')

    ## Generate sine table
    t = np.arange(512)
    sine = np.sin((2 * np.pi * t) / 512)
    #print(sine)

    assert(len(mag) == len(freq))

    for i in len(mag):

