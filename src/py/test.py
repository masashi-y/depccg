
import os

files = dict(enumerate(os.listdir(".")))
print "which file to use:"
for i, f in files.items():
    print "{}\t{}".format(i, f)
choice = raw_input()
print files[int(choice)]
