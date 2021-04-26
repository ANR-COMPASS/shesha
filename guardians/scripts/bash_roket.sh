#!/bin/bash
DIAM="8"
NSSP="40"
NFILT="20"
NPIX="4"
PIXSIZE="0.6"
WINDSPEED="10 20"
WINDDIR="0" # 45 90 135 180 20"
DATE=`date +%F_%Hh%M`
OUTPUT="$SHESHA_ROOT/guardians/scripts/outputfile_$DATE"

echo "writing output in "$OUTPUT

script="$SHESHA_ROOT/guardians/scripts/script_roket_fix.py"

for s in $WINDSPEED
do
    for dd in $WINDDIR
    do
        CMD="ipython $script $SHESHA_ROOT/guardians/script/Sim_param_r0_012.py -- --niter 200 --diam $DIAM --npix $NPIX --pixsize $PIXSIZE --nfilt $NFILT --nssp $NSSP --winddir $dd --windspeed $s -s roket_"$DIAM"m_nssp"$NSSP"_dir"$dd"_speed"$s".h5"
        echo "execute $CMD" >> $OUTPUT
        $CMD 2>> $OUTPUT >> $OUTPUT
    done
done
