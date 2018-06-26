Bebel
=====

A synthetic bicycle bell.

The goal of this project is to make a synthetic bike bell with the following
requirements:
- actually sounds like a bicycle bell
- louder than a physical bell and potentially lighter weight
- more reliable than a physical bell, always rings the same
- can be actuated from a number of switches from different places on the bars
- cheap hardware: target is Arduino with minimal external components
- sound may be changed (offline)

Design choices
==============

No PCM data is stored. Sound is synthesized on the fly. The DAC is 1-bit and
works like a class-D amplifier. The amplifier itself will be a transformer
driven by a mosfet. Consumption should be minimal in sleep, ideally even null.
If possible the bell switches act as power switches and the device goes
completely off after a predefined time.

TODO list
=========

- decompose a real sound bell in basic components: DONE analysis.py
- prototype synthesis: WIP gen.py
- synthesis in C: TODO
- benchmark synthesis on Arduino: TODO
- 1-bit DAC on Arduino: TODO
- prototype synthesys circuit: TODO
- prototype power on/wake up circuit: TODO
- make printable box: TODO
- make printable switches: TODO

