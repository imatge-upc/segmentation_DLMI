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

if [ "${PARAMS}" = "params_mask"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=15G  python train_mask.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_mask_seg"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=15G  python train_mask_seg.py -p"${PARAMS}"

elif [ "${PARAMS}" = "params_seg"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=15G  python train_seg.py -p"${PARAMS}"

elif [ "${PARAMS}" = "params_train"  ]
    then
    srun  -p fast --pty  --mem=15G  python train.py -p"${PARAMS}"

elif [ "${PARAMS}" = "params_survival"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=15G  python train_survival.py -p"${PARAMS}"
else
    echo 'ERROR in "train.sh"'
    echo 'Please, specify a valid parameter filename'
fi