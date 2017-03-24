
import sys
import depccg

model = "/home/masashi-y/tmp/models/tri_headfirst"
doc = [l.strip() for l in open(sys.argv[1])]

parser = depccg.PyAStarParser(model, loglevel=1)

res = parser.parse_doc(doc)
for i, r in enumerate(res):
    print "ID={}".format(i)
    print r.deriv
