import sys
fileName = sys.argv[1]

from jittor.utils.pytorch_converter import convert

with open(fileName, 'r') as f:
    lines = f.readlines()

code = "".join(lines)
print(code)
# print(type(code))
jittorCode = convert(code)

saveName = fileName.split('.py')[0] + "Jittor.py"

with open(saveName, 'w') as f:
    print(jittorCode, file=f)
