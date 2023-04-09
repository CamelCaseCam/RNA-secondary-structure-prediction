import ct2num
import numpy as np
import os
import matplotlib.pyplot as plt
import gc

dots = []
index = 0
os.makedirs("output_nums_-50", exist_ok=True)
for file in os.listdir('.'):
    if file.endswith('.ct'):
        index += 1
        _, dot = ct2num.ct2num(open(file).read())
        dots.append(dot)
        print('.', end='', flush=True)
        if index % 100 == 0:
            print('')
np.savez_compressed(f'output_nums/dots-50.npz', dots)
dots = []
gc.collect()