#!/bin/bash
#FILES="scao_sh_16x16_8pix.py "
# scao_sh_40x40_8pix.py  scao_sh_64x64_8pix.py
FILES_SCAO="scao_sh_16x16_8pix.py scao_sh_40x40_8pix.py scao_sh_80x80_8pix.py"
FILES_PYR="scao_pyrhr_16x16.py scao_pyrhr_40x40.py scao_pyrhr_80x80.py"
#FILES_MCAO="mcao_8m.py mcao_40m.py"
#FILES+="scao_sh_16x16_16pix.py"
# scao_sh_40x40_16pix.py scao_sh_64x64_16pix.py scao_sh_80x80_16pix.py"
#FILES+="scao_16x16_8pix_noisy.par scao_40x40_8pix_noisy.par scao_64x64_8pix_noisy.par scao_80x80_8pix_noisy.par
#FILES+="scao_16x16_16pix_noisy.par scao_40x40_16pix_noisy.par scao_64x64_16pix_noisy.par scao_80x80_16pix_noisy.par"
#FILES="scao_pyr_80x80_8pix.py"

DATE=`date +%F_%Hh%M`
OUTPUT="$SHESHA_ROOT/data/bench-results/outputfile_$DATE\_$HOSTNAME"
DEVICE=$1

echo "writing output in "$OUTPUT

script="$SHESHA_ROOT/test/benchmark_script.py"
DEVICE="0"
for f in $FILES_SCAO
do
    for CTR in "ls" "modopti" "mv" "geo"
    do
        for COG in "cog" "tcog" "wcog" "bpcog"
        do
            CMD="python $script $f $COG $CTR $DEVICE"
            echo "execute $CMD" >> $OUTPUT
            $CMD 2>> $OUTPUT >> $OUTPUT
        done
    done
done

# for f in $FILES_MCAO
# do
#     for CTR in "mv"
#     do
#         for COG in "cog"
#         do
#             CMD="python $script $f $COG $CTR $DEVICE"
#             echo "execute $CMD" >> $OUTPUT
#             $CMD 2>> $OUTPUT >> $OUTPUT
#         done
#     done
# done

DEVICE="0123"
for f in $FILES_PYR
do
    for CTR in "ls"
    do
        for COG in "pyr" #"pyr" #
        do
            CMD="python $script $f $COG $CTR $DEVICE"
            echo "execute $CMD" >> $OUTPUT
            $CMD 2>> $OUTPUT >> $OUTPUT
        done
    done
done



FILES_LGS="scao_sh_16x16_8pix_lgs.py"
#FILES_LGS+="scao_sh_40x40_10pix_lgs.par"
#FILES_LGS+="scao_sh_64x64_16pix_lgs.par"
#FILES_LGS+="scao_sh_80x80_20pix_lgs.par"
DEVICE="0"
for f in $FILES_LGS
do
    for CTR in "ls" "modopti" "mv" "geo"
    do
        for COG in "wcog" "corr"
        do
            CMD="python $script $f $COG $CTR $DEVICE"
            echo "execute $CMD" >> $OUTPUT
            $CMD 2>> $OUTPUT >> $OUTPUT
        done
    done
done
