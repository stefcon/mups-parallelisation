import sys
import re
from datetime import timedelta

rms_line = 'RMS absolute error in solution'
timestamp_re = r'\d{2}:\d{2}:\d{2}|$'


ACCURACY = 0.01
sum = 0
sum1 = 0
lines = []
flag = True
timestamp_flag = False
start_timestamp = None
end_timestamp = None
i = 0

def convert_to_seconds(timestamp):
    hh, mm, ss = timestamp.split(':')
    return int(timedelta(hours=int(hh),minutes=int(mm),seconds=int(ss)).total_seconds())

with open('../feyman/results.txt', 'a+') as f:
    rez = f.readlines()
    nl = len(rez)
    for line in sys.stdin:
        line = line.replace("\t", "")
        line = line.strip()

        if len(re.search(timestamp_re, line).group()) and start_timestamp is None:
            timestamp_flag = True
            start_timestamp = convert_to_seconds(re.search(timestamp_re, line).group())
        elif rms_line in line:
            line = line.split()
            error = float(line[-1])
            if nl == 0:
                f.write(f'{error}\n')
            else:
                reference = float(rez[i])
                if abs(reference - error) > ACCURACY:
                    flag = False
                    f.write(f"FAILED: {error} {reference} \n")
                i += 1
        elif len(re.search(timestamp_re, line).group()) and end_timestamp is None:
            timestamp_flag = False
            end_timestamp = convert_to_seconds(re.search(timestamp_re, line).group())
            # print(f"Start: {start_timestamp} End: {end_timestamp}")
            sum += end_timestamp - start_timestamp
            start_timestamp = end_timestamp = None
            # print("Cur time: " + str(sum))
        elif len(line)>0 and line[0] == 'T':
            line = line.split()
            sum1 += float(line[-1])

        
    if flag: print("Test PASSED")
    else: print("Test FAILED")

print("Time: " + str(sum1))
with open('time.txt', 'a+') as f:
    nl = len(f.readlines())
    f.write(str(sum1))
    if nl == 0:
        f.write(' seq')
    elif nl == 1: 
        f.write(' seq_omp')
    f.write('\n')
