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
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=15G  python train_VNET.py -p"${PARAMS}"
#    srun  -p fast --pty  --mem=20G  python train_VNET.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_VNET_1"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=15G  python train_VNET_1.py -p"${PARAMS}"
#    srun  -p fast --pty  --mem=20G  python train_VNET_1.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_VNET_full"  ]
    then
    srun  --gres=gpu:1,gmem:11GB --pty  --mem=15G  python train_VNET_full.py -p"${PARAMS}"
    elif [ "${PARAMS}" = "params_VNET_survival"  ]
    then
    srun  --gres=gpu:1,gmem:11GB --pty  --mem=15G  python train_VNET_survival.py -p"${PARAMS}"
#    srun  -p fast --pty  --mem=20G  python train_VNET_full.py -p"${PARAMS}"
else
    echo 'ERROR in "train.sh"'
    echo 'Please, specify a valid parameter filename'
fi