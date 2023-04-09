'''
This library converts a ct file with an RNA secondary structure to a numpy dot plot.
'''

import numpy as np

def ct2dot(ctfile: str, seqlen = 2048):
    words = ctfile.split()
    if seqlen == -1:
        seqlen = int(words[0])
    RecordName = words[1]
    del words
    ct = np.zeros((seqlen, seqlen))
    lines = ctfile.replace('\r', '').split('\n')[1:seqlen]
    for line in lines:
        if line == '':
            continue
        base = line.split()
        index = int(base[0]) - 1
        if base[4] != '0':
            pair = int(base[4]) - 1
            if pair >= seqlen:
                continue

            ct[index, pair] = 1
            ct[pair, index] = 1
    return RecordName, ct