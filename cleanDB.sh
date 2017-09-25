#!/bin/bash
 
while getopts ":f" opt; do
  case $opt in
    f)
      echo " Force removal HDF5 files in "$SHESHA_ROOT
      find $SHESHA_ROOT -name "*.h5" -exec rm {} \;
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 2
      ;;
  esac
done


echo "searching HDF5 files (will be removed)"
FIND=`find $SHESHA_ROOT -name "*.h5"`
if [[ -n $FIND ]]
then
  echo $FIND
  while true; do
    read -p "Do you ve to remove them? (y/n)" yn
    case $yn in	
        [Yy]* ) find $SHESHA_ROOT -name "*.h5" -exec rm {} \;; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
  done
else
  echo "nothing found!"
fi
