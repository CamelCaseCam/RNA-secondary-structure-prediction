import subprocess
import os

def LoadFile(FileName):
    file = subprocess.check_output("\"C:\Program Files\RNAstructure6.4\exe\ct2dot\" \"" + FileName + "\" 1 -", shell=True)
    lines = str(file).split("\\r\\n")
    Output = lines[1] + ","
    for i in range(len(lines[2])):
        if not lines[2][i] == '.' and not lines[2][i] == '(' and not lines[2][i] == ')':
            Output = Output + '.'
        else:
            Output = Output + lines[2][i]
    return Output

Dataset = "Sequence,Structure"
for file in os.listdir("D:\\Development\\RNAFolding\\archiveII\\"):
    if os.path.splitext(file)[1] == ".ct":
        Dataset = Dataset + "\n" + LoadFile(file)
f = open("Output.csv", 'w')
f.write(Dataset)