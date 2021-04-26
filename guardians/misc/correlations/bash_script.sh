#!/bin/bash

WINDDIR="0 45 90 135 180"
WINDSPEED="10 15 20"
GAIN="0.1 0.2 0.3 0.4"
OUTPUT="$SHESHA_ROOT/test/roket/correlations/outputfile"

echo "writing output in "$OUTPUT

script="$SHESHA_ROOT/test/roket/correlations/script_roket_cpu.py"

for D in $WINDDIR
do
    for S in $WINDSPEED
    do
      for G in $GAIN
      do

        CMD="ipython $script $D $S $G"
        echo "execute $CMD" >> $OUTPUT
        $CMD 2>> $OUTPUT >> $OUTPUT
      done
    done
done
