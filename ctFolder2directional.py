import ct2directional
import numpy as np
import os
import matplotlib.pyplot as plt
import gc


RNA_ENCODINGS = {
    'A': [1, 0, 0, 0, 1, 1],
    'T': [0, 1, 0, 0, 1, 1],
    'U': [0, 1, 0, 0, 1, 1],
    'C': [0, 0, 1, 0, 1, 1],
    'G': [0, 0, 0, 1, 1, 1],
    'x': [0, 0, 0, 0, 1, 1],
}

RNA_ENCODINGS_NPOS = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'U': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
    'x': [0, 0, 0, 0],
}

RNA_ENCODINGS_IDX = {
    'A': 1,
    'T': 2,
    'U': 2,
    'C': 3,
    'G': 4,
    'x': 0,
}

def EncodeRNA(RNA, npadlen):
    '''return np.array([
        np.multiply(
            RNA_ENCODINGS[RNA[base]], [1, 1, 1, 1, # identity
                ct2directional.ramp(base, 0, npadlen, ct2directional.MIN_VAL, ct2directional.MAX_VAL), # positional encoding from 0.5 to 1
                -ct2directional.ramp(base, 0, npadlen, ct2directional.MAX_VAL, ct2directional.MIN_VAL)] # positional encoding from -1 to -0.5
        )
        for base in range(len(RNA))
    ])'''
    return np.array([
        RNA_ENCODINGS_IDX[RNA[base]] for base in range(len(RNA))
    ])



dots = []
seqs = []
index = 0
os.makedirs("output_directional", exist_ok=True)
for file in os.listdir('.'):
    if file.endswith('.ct'):
        index += 1
        out = ct2directional.ct2directional(open(file).read())
        if out is None:
            continue
        _, dot, outseq, npadlen = out
        dots.append(dot)
        seqs.append(EncodeRNA(outseq, npadlen))
        print('.', end='', flush=True)
        if index % 100 == 0:
            print('')
np.savez_compressed(f'output_directional/dots.npz', dots)
np.savez_compressed(f'output_directional/seqs.npz', seqs)
dots = []
seqs = []
gc.collect()