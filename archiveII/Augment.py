import subprocess

def GenerateCandidate(StructureFile):
    Output = subprocess.check_output("RNAinverse -Fmp -f 0.5 -R5 -d2 ", shell=True, stdin=StructureFile, timeout=999)
    SecondaryStructure = open("struct.txt").read()
    OutputText = ""
    Output = str(Output).split("\\r\\n")
    for strand in Output:
        if strand == "'":
            continue
        if strand[0] == 'b':
            OutputText = OutputText  + strand.split(" ")[0].split("b'")[1]
        else:
            OutputText = OutputText + strand.split(" ")[0]
        OutputText = OutputText + "," + SecondaryStructure + "\n"
    return OutputText

f = open("Output.csv", 'a')
f.write(GenerateCandidate(open("struct.txt")))