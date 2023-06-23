import sys

sum = 0
lines = []
flag = 1
i = 0
with open('../prime/results.txt', 'a+') as f:
    rez = f.readlines()
    nl = len(rez)
    for line in sys.stdin:
        line = line.replace("\t", "")
        line = line.strip()
        if len(line)>0 and line[-1].isnumeric():
            line = line.split()
            if nl == 0:
                f.write(line[0] + " " + line[1] + "\n")
            else:
                if(line[0] + " " + line[1] + "\n" != rez[i]): 
                    flag = 0
                    f.write("FAILED: " + line[0] + " " + line[1] + "\n")
                i+=1
            sum += float(line[2])
    print("Time: " + str(sum))
    if flag: print("Test PASSED")
    else: print("Test FAILED")

with open('time.txt', 'a+') as f:
    nl = len(f.readlines())
    f.write(str(sum))
    if nl == 0:
        f.write(' seq')
    elif nl == 1: 
        f.write(' seq_omp')
    elif nl == 2:
        f.write(' seq_mod')
    elif nl == 3:
        f.write(' seq_mod_omp')
    f.write('\n')
