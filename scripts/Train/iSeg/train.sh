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
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=50G  python train_VNET.py -p"${PARAMS}"
#    srun  -p fast --pty  --mem=8G  python train_VNET.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_VNET_patches"  ]
    then
    srun  --gres=gpu:1,gmem:12GB --pty  --mem=20G  python train_VNET.py -p"${PARAMS}"
elif [ "${PARAMS}" = "params_SCAE"  ]
    then
    srun --gres=gpu:1,gmem:12GB --pty  --mem=6G  python train_SCAE.py -p"${PARAMS}"
    elif [ "${PARAMS}" = "params_VNET_ACNN"  ]
    then
    srun --gres=gpu:1,gmem:12GB --pty  --mem=6G  python train_VNET_ACNN.py -p"${PARAMS}"
else
    echo 'ERROR in "train.sh"'
    echo 'Please, specify a valid parameter filename'
fi