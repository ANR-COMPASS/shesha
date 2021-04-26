#!/bin/bash

# script="$SHESHA_ROOT/shesha/tests/check.py"
rm -f check.h5
script="$SHESHA_ROOT/tests/check.py"
conf_path="$SHESHA_ROOT/data/par/par4tests"
nb_test=$(ls -1 $conf_path/*.py | wc -l)
current_test=1
for file in $conf_path/*.py
do
    name=$(basename $file ".py")
    CMD="python $script $file"
    echo "[$current_test/$nb_test] running $name"
    $CMD > /dev/null
    # let "current_test++"
    current_test=$(expr $current_test + 1)
done
CMD="python $script osef --displayResult --repportResult=report_E2E.md"
$CMD
