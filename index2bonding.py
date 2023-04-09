import numpy as np


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

# Takes a numpy array of indices and returns a numpy array of bonding (negative if before, positive if after)
def EncodeSS(SecondaryStructure):
    Bonding = np.zeros(len(SecondaryStructure))
    for i in range(len(SecondaryStructure)):
        if SecondaryStructure[i] == -1:
            continue
        if SecondaryStructure[i] > i:
            Bonding[i] = ramp(SecondaryStructure[i], 0, len(SecondaryStructure) - 1, MIN_VAL, MAX_VAL)
        else:
            Bonding[i] = -ramp(SecondaryStructure[i], 0, len(SecondaryStructure) - 1, MAX_VAL, MIN_VAL)
    return Bonding


# now for decoding. This is going to be a bit more complicated, since we need to find the best match for each index
# A = 0, T = 1, G = 2, C = 3 
CAN_BOND = [
    [False, True, True, True],
    [True, True, True, False],
    [False, True, False, True],
    [False, False, True, False],
]


def MakeBondGroup(bases, struct, i):
    '''
    Makes a bond group starting at index i
    '''
    bond_group = []
    while i < len(struct) and struct[i] != 0:
        bond_group.append(i)
        i += 1
    return bond_group

def Decode(struct, bases):
    '''
    We need to to find the best match for each group of indices in the structure. We'll do this probabalistically using 
    dot plots. Steps below: (with b1 as base 1 and b2 as base 2)
    1. If abs(b1) or abs(b2) is at or near 0, they can't be bonded
    2. If sign(b1) == sign(b2), they can't be bonded
    3. If CAN_BOND[b1, b2] is False, they can't be bonded
    
    A base is part of a bond group if the diference between it and the next bond is close to (MAX_VAL - MIN_VAL) / len(struct). 
    This means that the group of bases bond to bases next to each other. 
    1. Find bond groups that are the same size, aproximitely the same magnitude, and have the opposite sign
    2. If there are only two groups, they are bonded
    3. If there are more than two groups, we need to find the best match. Follow the above steps to evaluate the match, in order 
    of sign similarity
    '''
    

if __name__ == "__main__":
    # gen test data
    struct = [7, 6, 5, -1, -1, 2, 1, 0] # AAAGGTTT
    print(EncodeSS(struct))