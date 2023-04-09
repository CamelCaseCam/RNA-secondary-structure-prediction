'''
Algorithm that uses a FFT of RNA structures to perform template folding on the RNA sequences. 
The algorithm uses hierarchical templates (sizes TEMPLATE_SIZE_1, TEMPLATE_SIZE_2, TEMPLATE_SIZE_3, etc). Each generation of 
templates is averaged with the previous generation, going from largest to smallest. 

Each template generation is the average of the structures coded for by the templates, multiplied by the (mean absolute error/MSE)
of the template. Instead of dividing by the number of templates, it's divided by the total error of the templates. 
'''

import numpy as np

NUM_GENERATIONS = 4
TEMPLATE_SIZES = [200, 100, 50, 25]
LOSS_FN = "SE"

def SetHyperparameters(num_generations, template_sizes, loss_fn):
    global NUM_GENERATIONS
    global TEMPLATE_SIZES
    global LOSS_FN
    NUM_GENERATIONS = num_generations
    TEMPLATE_SIZES = template_sizes
    LOSS_FN = loss_fn

def LoadTemplates(template_size):
    # Load the templates for the given template size
    arrs = np.load("Templates/Template_Size_" + str(template_size) + ".npz", allow_pickle=True)
    seqs = arrs["sequences"]
    templates = arrs["structures"]
    return zip(seqs, templates), len(templates)

TEMPLATES = []
NUM_TEMPLATES = []
def LoadAllTemplates():
    global TEMPLATES
    global NUM_TEMPLATES
    for x in TEMPLATE_SIZES:
        tmp = LoadTemplates(x)
        TEMPLATES.append(tmp[0])
        NUM_TEMPLATES.append(tmp[1])

DNA2idx = {
    "A": [1, 0, 0, 0],
    "a": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "t": [0, 1, 0, 0],
    "U": [0, 1, 0, 0],
    "u": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "g": [0, 0, 1, 0],
    "C": [0, 0, 0, 1],
    "c": [0, 0, 0, 1],
}

def DNA2Num(DNA):
    return np.array([DNA2idx[x] for x in DNA])

# funky average function
def SE(x, y):
    sqarr = np.abs((x - y))
    s = np.sum(sqarr, axis=-1)
    maxs = np.max(s)
    return np.reshape(s/np.max(s), (-1, 1)) ** 2 if maxs != 0 else np.reshape(np.zeros_like(s), (-1, 1))


def AE(x, y):
    arr = np.abs(x - y)
    d = np.sum(y != 0, axis=-1)
    id = np.divide(1, d, out=np.zeros_like(d, dtype=np.float64), where=d!=0)
    return np.reshape(np.sum(arr, axis=-1)*id, (-1, 1))

losses = {
    "SE": SE,
    "AE": AE
}




def Fold(DNA):
    numbases = len(DNA)
    # convert DNA to numerical representation
    DNA = DNA2Num(DNA)

    # reserve space for secondary structure
    totalstructure = np.zeros((numbases, 2), dtype=np.complex128)

    # loop through the generations of templates
    for i in range(NUM_GENERATIONS):
        structure = np.zeros((numbases, 2), dtype=np.complex128)
        # get the templates for the current generation
        templates = TEMPLATES[i]

        # get the template size for the current generation
        template_size = TEMPLATE_SIZES[i]

        tnum = NUM_TEMPLATES[i]

        # make sure the template size is less than the number of bases
        if template_size > numbases:
            continue

        # loop through the structure
        for j in range(numbases - template_size + 1):
            # Take the FFT of this section of the DNA sequence
            sequence = np.fft.fft(DNA[j:j+template_size])

            # Calculate the error of each template
            totalerror = np.zeros((template_size, 1))

            # loop through the templates
            for seq, template in templates:
                # Calculate the error of the template
                error = losses[LOSS_FN](sequence, seq)

                # Add the error to the total error
                totalerror += np.cast[np.float64](np.ones_like(error) - error)

                # Add the template structure to the structure
                structure[j:j+template_size] += template * (np.ones_like(error) - error)

            # Divide the structure by the total error
            inv_totalerror = np.divide(np.ones_like(totalerror), totalerror, out=np.full_like(totalerror, tnum, dtype=np.float64), where=totalerror!=0)
            structure[j:j+template_size] *= inv_totalerror
            print(f"Generation {i+1}/{NUM_GENERATIONS}", end="\r")
        # Average the structure with the previous generation
        totalstructure = (totalstructure + structure) / 2

    # return the structure
    return totalstructure


if __name__ == "__main__":
    DNA = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"
    EXPECTED_STRUCTURE = "(((((((..((((.........)))).(((((.......))))).....(((((.......))))))))))))."
    LoadAllTemplates()
    print("Templates loaded")
    folded = Fold(DNA)
    print(folded)
    for base in folded:
        if base[1] == 0:
            print('.', end='')
        else:
            if base[0] > 0:
                print('(', end='')
            else:
                print(')', end='')