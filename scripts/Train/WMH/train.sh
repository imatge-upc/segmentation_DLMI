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

if [ "${PARAMS}" = "params_full"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=20G  python train_VNET.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_patches"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=20G  python train_VNET.py -p"${PARAMS}"
else
    echo 'ERROR in "train.sh"'
    echo 'Please, specify a valid parameter filename'
fi