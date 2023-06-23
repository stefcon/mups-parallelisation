import sys

sum = 0
lines = []
flag = 1
i = 0
with open('../MolDyn/results.txt', 'a+') as f:
    rez = f.readlines()
    nl = len(rez)
    for line in sys.stdin:
        line = line.replace("\t", "")
        line = line.strip()
        if len(line)>0 and line[0].isnumeric():
            if nl == 0:
                f.write(line + "\n")
            else:
                if(line + "\n" != rez[i]): 
                    flag = 0
                    f.write("FAILED: " + line + "\n")
                i+=1
        elif len(line)>0 and line[0] == 'T':
            line = line.split()
            print("Time: " + line[-1])
            sum = float(line[-1])
    if flag: print("Test PASSED")
    else: print("Test FAILED")

with open('time.txt', 'a+') as f:
    nl = len(f.readlines())
    f.write(str(sum))
    f.write('\n')
