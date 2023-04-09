import ct2dot
import numpy as np
import os
import matplotlib.pyplot as plt
import gc

dots = []
index = 0
os.makedirs("output_1024", exist_ok=True)
for file in os.listdir('.'):
    if file.endswith('.ct'):
        index += 1
        _, dot = ct2dot.ct2dot(open(file).read(), seqlen=1024)
        dots.append(dot)
        print('.', end='', flush=True)
        if index % 100 == 0:
            print('')
        if index % 128 == 0:
            np.savez_compressed(f'output_1024/dots_{index / 128}.npz', dots)
            dots = []
            gc.collect()
np.savez_compressed(f'output_1024/dots_{int(index / 128) + 1}.npz', dots)
dots = []
gc.collect()