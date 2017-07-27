#!/usr/bin/env bash

for i in "$@"
do
case $i in
    -p=*|--params=*)
    PARAMS="${i#*=}"
    shift # past argument=value
    ;;
    *)
            # unknown option
    ;;
esac
done

if [ "${PARAMS}" = "params_VNET"  ]
    then
#    srun  --pty  --mem=8G  python test_VNET.py -p"${PARAMS}"
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=8G  python test_VNET.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_VNET_patches"  ]
    then
#    srun  --pty  --mem=8G  python test_VNET.py -p"${PARAMS}"
    srun  -p fast --gres=gpu:1,gmem:12GB --pty  --mem=8G  python test_VNET.py -p"${PARAMS}"
else
    echo 'ERROR in "train.sh"'
    echo 'Please, specify a valid parameter filename'
fi