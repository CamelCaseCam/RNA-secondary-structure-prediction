'''
Generates a set of templates for the template size given from the data. Templates consist of an FFT of the 
RNA sequence and an FFT of the secondary structure.
'''

import numpy as np
import os



def GenerateTemplates(data, template_size):
    '''
    Data is a list of tuples of the form (RNA sequence, secondary structure). RNA sequence is a 1-hot encoded numpy array
    of shape (sequence length, 4). Secondary structure is a array of size (sequence length, 2) where channel 1 represents the
    bonded nucleotide and channel 2 represents whether the nucleotide is bonded or not. 
    '''

    # Initialize the templates
    sequences = []
    structures = []

    # loop through the data and add the sequences and structures to the templates
    for RNA, struct in data:
        # break the RNA sequence into overlapping chunks of size template_size
        for i in range(0, len(RNA) - template_size + 1):
            # take the FFT of the RNA sequence and the secondary structure
            sequence = np.fft.fft(RNA[i:i+template_size])
            structure = np.fft.fft(struct[i:i+template_size])
        
            # add the sequence and structure to the templates
            sequences.append(sequence)
            structures.append(structure)
    
    # convert the templates to numpy arrays
    sequences = np.array(sequences, dtype=np.complex128)
    structures = np.array(structures, dtype=np.complex128)

    return sequences, structures

def ramp(value, sourcemin, sourcemax, destmin, destmax):
    # Figure out how 'wide' each range is
    leftSpan = sourcemax - sourcemin
    rightSpan = destmax - destmin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - sourcemin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return destmin + (valueScaled * rightSpan)

def loadct(ctfile):
    words = ctfile.split()
    seqlen = int(words[0])
    RecordName = words[1]
    del words

    ct = np.zeros((seqlen, 2))
    lines = ctfile.replace('\r', '').split('\n')[1:seqlen]
    outseq = np.zeros((seqlen, 4))
    for line in lines:
        if line == '':
            continue
        base = line.split()
        index = int(base[0]) - 1
        outseq[index, "ACGU".index(base[1])] = 1

        if base[4] != '0':
            pair = int(base[4]) - 1
            if index < pair:
                ct[index, 0] = abs(index-pair)/seqlen
                ct[index, 1] = 1
            else:
                ct[index, 0] = -abs(index-pair)/seqlen
                ct[index, 1] = 1
    return RecordName, outseq, ct, seqlen

def GetDataFromFolder(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".ct"):
            with open(os.path.join(folder, file), "r") as f:
                ct = f.read()
                ct = loadct(ct)
                data.append([ct[1], ct[2]])
                assert len(ct[1]) == len(ct[2])
                print(".", end="")
    for x, y, in data:
        try:
            assert len(x) == len(y)
        except:
            print(len(x), len(y))
            raise
    return data

def GenerateTemplatesFromPath(data, template_size):
    '''
    Loads the data from the given path and generates the templates.
    '''
    templates = GenerateTemplates(data, template_size)

    os.makedirs(os.path.join("Templates"), exist_ok=True)

    np.savez_compressed(f"Templates/Template_Size_{template_size}", sequences=templates[0], structures=templates[1])


GENERATE_TEMPLATES = False

if __name__ == "__main__":

    data = GetDataFromFolder("D:\\Development\\RNAFolding\\archiveII")

    print("\n\nLoaded data!")

    GenerateTemplatesFromPath(data, 200)
    print("\n\n========================================\n\n")
    GenerateTemplatesFromPath(data, 100)
    print("\n\n========================================\n\n")
    GenerateTemplatesFromPath(data, 50)
    print("\n\n========================================\n\n")
    GenerateTemplatesFromPath(data, 25)

    print("\n\nDone!")