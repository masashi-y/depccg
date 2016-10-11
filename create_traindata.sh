#!/bin/sh

if [ $# -ne 2 ]; then
    echo "Wrong number of arguments."
    echo "./create_traindata.sh PATH_TO_AUTO_DIR OUTPUT_DIR"
    exit 1
fi

AUTODIR=$1
OUTDIR=$2

iter=1
for i in `seq 2 21`; do
    if [ $i -lt 10 ]; then i=0$i; fi
    for auto in `find "$AUTODIR/$i" -name "*.auto"`; do
        echo $auto
        python tagger.py --create --outdir $OUTDIR $auto &
        if (($iter % 40 == 0)); then wait; fi
        iter=$((iter + 1))
    done
done
