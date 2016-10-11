#!/bin/sh

if [ $# -ne 3 ]; then
    echo "Wrong number of arguments."
    echo "./create_traindata.sh {train,val,test} PATH_TO_AUTO_DIR OUTPUT_DIR"
    exit 1
fi

if [ $1 = "train" ]; then
    SUBSET=`seq 2 21`
elif [ $1 = "val" ]; then
    SUBSET=0
elif [ $1 = "test" ]; then
    SUBSET=23
else
    echo "Wrong choice for subset: $1"
    echo "Choice must be makde among {train,val,test}"
    exit 1
fi
    
AUTODIR=$2
OUTDIR=$3

iter=1
for i in $SUBSET; do
    if [ $i -lt 10 ]; then i=0$i; fi
    for auto in `find "$AUTODIR/$i" -name "*.auto"`; do
        echo $auto
        python tagger.py --create --outdir $OUTDIR $auto &
        if (($iter % 40 == 0)); then wait; fi
        iter=$((iter + 1))
    done
done
