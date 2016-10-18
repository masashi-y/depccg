
from ccgbank import *

a = AutoReader("/home/cl/masashi-y/ccgbank_1_1/data/AUTO/00/wsj_0001.auto").readall()

for line in a:
    print line
