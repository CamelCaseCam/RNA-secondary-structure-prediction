# converts a ct file to a directional encoding

MAX_VAL = 1
MIN_VAL = 0.5

def ramp(value, sourcemin, sourcemax, destmin, destmax):
    # Figure out how 'wide' each range is
    leftSpan = sourcemax - sourcemin
    rightSpan = destmax - destmin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - sourcemin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return destmin + (valueScaled * rightSpan)

import numpy as np

def ct2directional(ctfile: str, seqlen = 2048):
    words = ctfile.split()
    if seqlen == -1:
        seqlen = int(words[0])
    npadlen = int(words[0])
    if int(words[0]) > seqlen:
        return None
    RecordName = words[1]
    del words
    ct = np.full(seqlen, 3.0)
    lines = ctfile.replace('\r', '').split('\n')[1:seqlen]
    outseq = ["x"] * seqlen
    for line in lines:
        if line == '':
            continue
        base = line.split()
        index = int(base[0]) - 1
        outseq[index] = base[1]

        if base[4] != '0':
            pair = int(base[4]) - 1
            if index < pair:
                ct[index] = ramp(pair, 0, npadlen, MIN_VAL, MAX_VAL)
            else:
                ct[index] = -ramp(pair, 0, npadlen, MAX_VAL, MIN_VAL)
    return RecordName, ct, outseq, npadlen